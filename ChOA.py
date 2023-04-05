#导入库
import numpy as np
import matplotlib.pyplot as plt

#混沌映射
def logistics_chaotic_map(x):
    return 4*x*(1-x)



#社会行为刺激行为模型
def social_incentive(chimp, prey, miu ,a, d):
    if miu<0.5:
        return prey - a * d
    else:
        return logistics_chaotic_map(chimp)
    


#实现黑猩猩算法
def CHOA(fobj, lb, ub, dim, N, Group_num, Max_iter):
    
    #初始化种群位置
    Groups = np.zeros((Group_num, N, dim))
    for i in range(Group_num):
        Groups[i] = np.random.uniform(0, 1, (N, dim)) * (ub - lb) + lb
        
        
        #初始化f
        f = np.zeros((Group_num,1)) + 2.5

    #初始化m,a和c
    m = np.random.uniform(0, 1,(Group_num, N))
    r1 = np.random.uniform(0, 1,(Group_num, N))
    r2 = np.random.uniform(0, 1,(Group_num, N))
    a = 2 * f * r1  -  f
    c = 2 * r2


    #计算每只黑猩猩的适应度
    fitness = fobj(Groups)

    #初始化attacker、chaser、barrier和driver
    best_index = np.argsort(fitness.reshape(Group_num * N))
    four_chimp = best_index[-4:][::-1]
    three_index = np.zeros((4, 2), dtype=np.int)
    for i in range(4):
        three_index[i,0] = four_chimp[i] // (N * dim)
        three_index[i,1] = four_chimp[i] % (N * dim) // dim
    attacker = Groups[three_index[0][0], three_index[0][1]]
    chaser = Groups[three_index[1][0], three_index[1][1]]
    barrier = Groups[three_index[2][0], three_index[2][1]]
    driver = Groups[three_index[3][0], three_index[3][1]]

    #初始化猎物的位置
    prey = (attacker + chaser + barrier + driver) / 4

    #初始化种群和猎物的距离
    distance = np.zeros((Group_num, N, dim))
    for i in range(Group_num):
        for j in range(N):
            distance[i,j] = np.abs(c[i,j] * prey - m[i,j] * Groups[i,j])

    #迭代
    for iter in range(Max_iter):
        miu = np.random.uniform(0, 1,(Group_num, N))

        f[0,:] = 2.5 - 2 * np.log(iter + 1e-8) / np.log(Max_iter)
        f[1,:] = -2 * iter**3 / Max_iter**3 +2.5
        f[2,:] = 0.5 + 2 * np.exp(-(4 * iter / Max_iter)**2)
        f[3,:] = 2.5 + 2 * (iter/Max_iter)**2 - 2 * (2 * iter / Max_iter)
        m = logistics_chaotic_map(m)

        if iter != 0:
            r1 = np.random.uniform(0, 1,(Group_num, N))
            r2 = np.random.uniform(0, 1,(Group_num, N))
            a = 2 * r1 * f -  f
            c = 2 * r2

        for i in range(Group_num):
            for j in range(N):
                Groups[i,j] = social_incentive(chimp = Groups[i,j], prey = prey, miu = miu[i,j], a = a[i,j], d = distance[i,j])
                Groups[i,j] = np.clip(Groups[i,j], lb, ub)

        if np.max(np.abs(a)) <= 1:  #搜索阶段--调整attacker、chaser、barrier和driver的个体
            fitness = fobj(Groups)

            #更新attacker、chaser、barrier和driver
            best_index = np.argsort(fitness.reshape(Group_num * N))
            four_chimp = best_index[-4::-1]
            three_index = np.zeros((4, 2), dtype=np.int)
            for i in range(4):
                three_index[i,0] = four_chimp[i] // (N * dim)
                three_index[i,1] = four_chimp[i] % (N * dim) // dim


            attacker = Groups[three_index[0][0], three_index[0][1]]
            a1, c1 ,m1 = a[three_index[0][0], three_index[0][1]], c[three_index[0][0], three_index[0][1]], m[three_index[0][0], three_index[0][1]]
            chaser = Groups[three_index[1][0], three_index[1][1]]
            a2, c2 ,m2 = a[three_index[1][0], three_index[1][1]], c[three_index[1][0], three_index[1][1]], m[three_index[1][0], three_index[1][1]]
            barrier = Groups[three_index[2][0], three_index[2][1]]
            a3, c3 ,m3 = a[three_index[2][0], three_index[2][1]], c[three_index[2][0], three_index[2][1]], m[three_index[2][0], three_index[2][1]]
            driver = Groups[three_index[3][0], three_index[3][1]]
            a4, c4 ,m4 = a[three_index[3][0], three_index[3][1]], c[three_index[3][0], three_index[3][1]], m[three_index[3][0], three_index[3][1]]

            d_attacker = np.abs(c1 * prey - m1 * attacker)
            d_chaser = np.abs(c2 * prey - m2 * chaser)
            d_barrier = np.abs(c3 * prey - m3 * barrier)
            d_driver = np.abs(c4 * prey - m4 * driver)

            x1 = attacker - a1 * d_attacker
            x3 = chaser - a2 * d_chaser
            x2 = barrier - a3 * d_barrier
            x4 = driver - a4 * d_driver

            prey = (x1 + x2 + x3 + x4) / 4

        else:  #捕猎阶段--更新attack、chase、barrier、driver和猎物的位置
            attacker = Groups[three_index[0][0], three_index[0][1]]
            a1, c1 ,m1 = a[three_index[0][0], three_index[0][1]], c[three_index[0][0], three_index[0][1]], m[three_index[0][0], three_index[0][1]]
            chaser = Groups[three_index[1][0], three_index[1][1]]
            a2, c2 ,m2 = a[three_index[1][0], three_index[1][1]], c[three_index[1][0], three_index[1][1]], m[three_index[1][0], three_index[1][1]]
            barrier = Groups[three_index[2][0], three_index[2][1]]
            a3, c3 ,m3 = a[three_index[2][0], three_index[2][1]], c[three_index[2][0], three_index[2][1]], m[three_index[2][0], three_index[2][1]]
            driver = Groups[three_index[3][0], three_index[3][1]]
            a4, c4 ,m4 = a[three_index[3][0], three_index[3][1]], c[three_index[3][0], three_index[3][1]], m[three_index[3][0], three_index[3][1]]
            
            d_attacker = np.abs(c1 * prey - m1 * attacker)
            d_chaser = np.abs(c2 * prey - m2 * chaser)
            d_barrier = np.abs(c3 * prey - m3 * barrier)
            d_driver = np.abs(c4 * prey - m4 * driver)

            x1 = attacker - a1 * d_attacker
            x3 = chaser - a2 * d_chaser
            x2 = barrier - a3 * d_barrier
            x4 = driver - a4 * d_driver

            prey = (x1 + x2 + x3 + x4) / 4


    #迭代完成返回优化结果
    return attacker, fobj(attacker)



                
    
