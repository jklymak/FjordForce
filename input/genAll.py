import gendata

num = 1
for wind in [30, 20, 15, 10]:
    for NsqFac in [0.5, 1, 2]:
        if ((NsqFac == 1) and (wind == 20)):
            print(f'Already Created Nsq={NsqFac}, wind={wind}')
            # this is run50
            continue
        num += 1
        runnum = 50 + num
        with open('./genall_runs.txt', 'a') as fout:
            fout.write(f'Bute3d{runnum}: wind={wind} NsqFac={NsqFac}\n')
        gendata.gendata(runnumber=runnum, wind=wind, NsqFac=NsqFac)


