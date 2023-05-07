from argparse import ArgumentParser
from enum import Enum
from redis import Connection, from_url
from redis.commands.search.field import VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
import numpy as np
from enum_actions import enum_action
from multiprocessing import Pool, cpu_count
from itertools import repeat
import uuid
from time import perf_counter, sleep

REDIS_URL: str = 'redis://default:redis@localhost:12000'
NUM_KEYS: int = 100000
connection: Connection = None

class OBJECT_TYPE(Enum):
    HASH = 'hash'
    JSON = 'json'

class INDEX_TYPE(Enum):
    FLAT = 'flat'
    HNSW = 'hnsw'

class METRIC_TYPE(Enum):
    L2 = 'l2'
    IP = 'ip'
    COSINE = 'cosine'

class FLOAT_TYPE(Enum):
    F32 = 'float32'
    F64 = 'float64'

class IndexTest(object):
    def __init__(self, args: dict):
        global connection
        connection = from_url(args.url)
        self.num_keys: int = args.nkeys
        self.object_type: OBJECT_TYPE = args.objecttype
        self.index_type: INDEX_TYPE = args.indextype
        self.metric_type: METRIC_TYPE = args.metrictype
        self.float_type: FLOAT_TYPE = args.floattype
        self.vec_dim: int = args.vecdim
        self.vec_m: int = args.vecm

    def test(self) -> dict:
        """ Public function for initiating all index tests.

            Returns
            -------
            None
        """
        global connection
        num_workers: int = cpu_count() - 1  #number of worker processes for data loading
        keys: list[int] = [self.num_keys // num_workers for i in range(num_workers)]  #number of keys each worker will generate
        keys[0] += self.num_keys % num_workers

        connection.flushdb()
        sleep(5) #wait for counters to reset
        base_ram: float = round(connection.info('memory')['used_memory']/1048576, 2)  # 'empty' Redis db memory usage
        vec_params: dict = {
            "TYPE": self.float_type.value, 
            "DIM": self.vec_dim, 
            "DISTANCE_METRIC": self.metric_type.value,
        }
        if self.index_type is INDEX_TYPE.HNSW:
            vec_params['M'] = self.vec_m
        
        match self.object_type:
            case OBJECT_TYPE.JSON:
                schema = [ VectorField('$.vector', self.index_type.value, vec_params, as_name='vector')]
                idx_def: IndexDefinition = IndexDefinition(index_type=IndexType.JSON, prefix=['key:'])
            case OBJECT_TYPE.HASH:
                schema = [ VectorField('vector', self.index_type.value, vec_params)]
                idx_def: IndexDefinition = IndexDefinition(index_type=IndexType.HASH, prefix=['key:'])
        connection.ft('idx').create_index(schema, definition=idx_def)
        pool_params = zip(keys, repeat(self.object_type), repeat(self.vec_dim), repeat(self.float_type))

        t1_start: float = perf_counter()
        with Pool(cpu_count()) as pool:
            pool.starmap(load_db, pool_params)  # load a Redis instance via a pool of worker processes
        t1_stop:float = perf_counter()

        sleep(5) # wait for counters to reset
        idx_info: dict = connection.ft('idx').info()   # calculate the memory usage of the index alone, MB
        index_ram: float = round(float(idx_info['inverted_sz_mb']) + \
            float(idx_info['vector_index_sz_mb']) + \
            float(idx_info['offset_vectors_sz_mb']) + \
            float(idx_info['doc_table_size_mb']) + \
            float(idx_info['sortable_values_size_mb']) + \
            float(idx_info['key_table_size_mb']), 2)
        tot_ram: float = round(connection.info('memory')['used_memory']/1048576, 2) # get the total memory usage of Redis instance, MB
        #print(f'tot_ram:{tot_ram}, index_ram:{index_ram}, base_ram:{base_ram}')
        data_ram: float = max(1,round(tot_ram - index_ram - base_ram, 2))  # calculate the data usage alone by substracting base and index
        ratio: float = round(index_ram/data_ram * 100, 2)  # calculate the index to data memory ratio
        sample_key: str = connection.randomkey()
        doc_size: int = connection.memory_usage(sample_key)  # calculate size of 1 document
        exec_time: float = round(t1_stop - t1_start, 2)
        return {'index_ram': index_ram, 
                'data_ram': data_ram, 
                'index_data': ratio, 
                'doc_size': doc_size,
                'exec_time': exec_time
        }

def load_db(keys: int, object_type: OBJECT_TYPE, vec_dim: int, float_type: FLOAT_TYPE) -> None:
    """ Function for loading vectors into Redis.  Runs in a multiprocess pool
        Parameters
        ----------
        key - number of keys to be generated
        vec_dim - dimension of each vector
        float_type - float type of vector (float32 or float64)
        Returns
        -------
        None
    """
    global connection
    for i in range(keys):   # loop creating hash or JSON objects with 1 field - a vector of floats
        vec: np.ndarray = np.random.default_rng().uniform(0.0, 5.0, size=vec_dim)
        match object_type:
            case OBJECT_TYPE.JSON:
                match float_type:
                    case FLOAT_TYPE.F32:
                        vec = vec.astype(np.float32).tolist()
                    case FLOAT_TYPE.F64:
                        vec = vec.astype(np.float64).tolist()
                connection.json().set(f'key:{str(uuid.uuid4())}', '$', {'vector': vec})
            case OBJECT_TYPE.HASH:
                match float_type:
                    case FLOAT_TYPE.F32:
                        vec = vec.astype(np.float32).tobytes()
                    case FLOAT_TYPE.F64:
                        vec = vec.astype(np.float64).tobytes()
                connection.hset(f'key:{str(uuid.uuid4())}', mapping={'vector': vec})

if __name__ == '__main__':
    parser = ArgumentParser(description='VSS Index Size Test')
    parser.add_argument('--url', required=False, type=str, default=REDIS_URL,
        help='URL connect string')
    parser.add_argument('--nkeys', required=False, type=int, default=NUM_KEYS,
        help='Number of keys to be generated')
    parser.add_argument('--objecttype', required=False, 
        action=enum_action(OBJECT_TYPE), default=OBJECT_TYPE.JSON,
        help='Redis Object Type')
    parser.add_argument('--indextype', required=False,
        action=enum_action(INDEX_TYPE), default=INDEX_TYPE.FLAT,
        help='Vector Index Type')
    parser.add_argument('--metrictype', required=False,
        action=enum_action(METRIC_TYPE), default=METRIC_TYPE.L2,
        help='Vector Metric Type')
    parser.add_argument('--floattype', required=False,
        action=enum_action(FLOAT_TYPE), default=FLOAT_TYPE.F32,
        help='Vector Float Type')
    parser.add_argument('--vecdim', required=False, type=int, default=1536,
        help='Vector Dimension')
    parser.add_argument('--vecm', required=False, type=int, default=16,
        help='HNSW M Param')
    args = parser.parse_args()

    tester: IndexTest = IndexTest(args)
    result = tester.test()
    print('Vector Index Test')
    print(' ')
    print('*** Parameters ***')
    print(f'nkeys: {args.nkeys}')
    print(f'objecttype: {args.objecttype.value}') 
    print(f'indextype: {args.indextype.value}')
    print(f'metrictype: {args.metrictype.value}')
    print(f'floattype: {args.floattype.value}')
    print(f'vecdim: {args.vecdim}')
    if (args.indextype is INDEX_TYPE.HNSW):
        print(f'vecm: {args.vecm}')
    print(' ')
    print('*** Results ***')
    print(f"index ram used: {result['index_ram']} MB")
    print(f"data ram used: {result['data_ram']} MB")
    print(f"index to data ratio: {result['index_data']}%")
    print(f"document size: {result['doc_size']} B")
    print(f"execution time: {result['exec_time']} sec")