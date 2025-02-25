import blosc2
import multiprocessing

def bl2_save(array, name, path):
    with open(f'{path}/{name}.bl2', 'wb') as f:
        f.write(blosc2.save_tensor(array))

def bl2_load(name, path):
    with open(f'{path}/{name}.bl2', 'rb') as f:
        return blosc2.load_tensor(f.read())

def process_results(results):
    # placeholder for now
    path = 'E:/vbn_s3_cache/visual-behavior-neuropixels-0.5.0/behavior_ecephys_sessions/'
    for result in results:
        if result is not None:
            print("Saving", result[0])
            bl2_save(result[1], result[0], path)

def multi(data, func):
    with multiprocessing.Pool() as pool:
        results = pool.map(func, data)
    process_results(results)