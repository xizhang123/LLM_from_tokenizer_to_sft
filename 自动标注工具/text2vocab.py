import time
import numpy as np
from tqdm import tqdm
import multiprocessing
import concurrent.futures
from collections import Counter,defaultdict

def get_n_grams(task):
    global ignored_set
    n,temp_str = task
    old_counter = set()
    new_counter = set()
    ret_counter = defaultdict(lambda: 1)
    for i in range(len(temp_str)-n+1):
        substring = temp_str[i:i + n]
        if len(ignored_set & set(substring)) == 0:
            if substring in ret_counter or substring in new_counter or substring in old_counter:
                ret_counter[substring] += 1
            elif len(new_counter) < 2e6/n:
                new_counter.add(substring)
            else:#存满的时候释放空间
                old_counter,new_counter = new_counter,set()
    return dict(ret_counter)

#参数设置
def get_vocab(N,ignored,merge_str,workers):
    global ignored_set
    #临时变量
    n_gram_frep_counter_dicts = {}
    #生成并统计全部片段
    ignored_set = set(ignored)
    for n in range(1, N+1):
        temp_counter = Counter()#元素不存在时自动设为0
        #耗时代码多线程加速
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            trunk_size = int(1e7)
            trunks = [(start,start+trunk_size) for start in range(0,len(merge_str),trunk_size)]
            tasks = [executor.submit(get_n_grams,[n,merge_str[start:end]]) for start,end in trunks[:workers]]
            trunks = trunks[len(tasks):]
            new_tasks = []
            with tqdm(total=len(trunks)+len(tasks), desc='['+str(n)+'/'+str(N)+']') as pbar:
                while True:
                    for future in concurrent.futures.as_completed(tasks):
                        temp_counter.update(future.result())
                        pbar.update(1)  # 每完成一个任务，更新进度条
                        if trunks:#完成一个提交一个
                            start,end = trunks[0]
                            trunks = trunks[1:]
                            if workers != 1:#将各进程错开
                                time.sleep(0.5)
                            new_tasks += [executor.submit(get_n_grams,[n,merge_str[start:end]])]
                    tasks,new_tasks = new_tasks,[]
                    if not tasks:
                        break
        n_gram_frep_counter_dicts[n] = temp_counter
    return n_gram_frep_counter_dicts
