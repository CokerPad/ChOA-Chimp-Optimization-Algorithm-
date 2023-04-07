#导入库
import numpy as np
import matplotlib.pyplot as plt

#混沌映射
def logistics_chaotic_map(dim, iteration=10, value=1):
    x0 = np.zeros(dim) + 0.7
    for i in range(iteration):
        x0 = 4 * x0 * (1 - x0)
    return x0



#社会行为刺激行为模型---还没用上
def social_incentive(chimp, prey, miu ,a, d):
    if miu<0.5:
        return prey - a * d
    else:
        return logistics_chaotic_map(chimp.shape[0])
    


#实现黑猩猩算法
def CHOA(fobj, lb, ub, dim, N, Max_iter):
    
    #初始化种群位置
    Groups = np.zeros((N, dim))
    Groups = np.random.uniform(0, 1, (N, dim)) * (ub - lb) + lb


    #计算每只黑猩猩的适应度
    fitness = fobj(Groups)



    #初始化f
    f = np.zeros((4, dim))

    #迭代
    for iter in range(Max_iter):

        #初始化f值
        f[0,:] = 2.5 - 2 * np.log(iter + 1e-8) / np.log(Max_iter)
        f[1,:] = -2 * iter**3 / Max_iter**3 +2.5
        f[2,:] = 0.5 + 2 * np.exp(-(4 * iter / Max_iter)**2)
        f[3,:] = 2.5 + 2 * (iter/Max_iter)**2 - 2 * (2 * iter / Max_iter)
    

        #计算适应度
        fitness = fobj(Groups)
        
        #初始化attacker、chaser、barrier和driver
        best_index = np.argsort(fitness.reshape(N))
        four_chimp_position = best_index[-4:][::-1]
        attacker_position = Groups[four_chimp_position[0]]
        chaser_position = Groups[four_chimp_position[1]]
        barrier_position = Groups[four_chimp_position[2]]
        driver_position = Groups[four_chimp_position[3]]

        for i in range(N):
            #计算a、f、c、m
            r1 = np.random.uniform(0, 1, (4, dim))
            r2 = np.random.uniform(0, 1, (4, dim))
            a = 2 * f * r1 - f
            c = 2 * r2
            m = logistics_chaotic_map(1)
            d_attacker = np.abs(c[0,:] * attacker_position - m * Groups[i])
            d_chaser = np.abs(c[1,:] * chaser_position - m * Groups[i])
            d_barrier = np.abs(c[2,:] * barrier_position - m * Groups[i])
            d_driver = np.abs(c[3,:] * driver_position - m * Groups[i])

            x1 = attacker_position - a[0,:] * d_attacker
            x3 = chaser_position - a[1,:] * d_chaser
            x2 = barrier_position - a[2,:] * d_barrier
            x4 = driver_position - a[3,:] * d_driver

            Groups[i] = np.clip((x1 + x2 + x3 + x4) / 4, lb, ub)



    #迭代完成返回优化结果
    return attacker_position, fobj(attacker_position)



                
    
