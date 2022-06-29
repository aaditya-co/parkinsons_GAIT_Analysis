dem=0
while dem<500:

    for x in range(1):
        i = random.randrange(0, min(len(testX), len(fog_testX)))
        #copy textX in new array
        pd_data = testX[i]
        pd_class = testy[i]
        fog_data = fog_testX[i]
        fog_class = fog_testy[i]
    
    #convert to list
    pd_data = pd_data.tolist()
    fog_data = fog_data.tolist()
    pd_class = pd_class.tolist()
    fog_class = fog_class.tolist()


    #save the data in text file seperated by comma
    with open('pd_data.txt', 'w') as f:
        for i in range(len(pd_data)):
            f.write(str(pd_data[i]) + ',' )
    with open('fog_data.txt', 'w') as f:
        for i in range(len(fog_data)):
            f.write(str(fog_data[i]) + ',' )
    with open('pd_class.txt', 'w') as f:
        for i in range(len(pd_class)):
            f.write(str(pd_class) + ',' )
    with open('fog_class.txt', 'w') as f:
        for i in range(len(fog_class)):
            f.write(str(fog_class) + ',' )


    if np.argmax(pd_class) == 0:
        folder = "pd/"
        print("Parkinson's Disease")
    else:
        folder = "Healthy/"
        print("Control")


    import os
    import math
    def fun(folder):
        last_nm = 0
        dir_name = 'check_data/' + folder
        list_of_files = sorted(filter( lambda x: os.path.isdir(os.path.join(dir_name, x)),os.listdir(dir_name) ) )
        for names in list_of_files:
            last_nm = max(last_nm,int(names[1::]))
        ans = last_nm+1
        return str(ans)

    folder_name = 'p' + fun(folder)
    final_nm = "check_data/"+ folder + folder_name
    os.mkdir(final_nm)


    #python program to move all text files
    import shutil
    files = ['pd_data.txt', 'fog_data.txt', 'pd_class.txt', 'fog_class.txt']
    destination = final_nm
    for source in files:
        shutil.move(source, destination)

    dem+= 1