#!/bin/bash

IFS=$'\n' # make newlines the only separator
# obs_dirs=/Volumes/SeaJade\ 2\ Backup/NZ/mseed_45_EH-BN/2020
# write_dir=/Volumes/SeaJade\ 2\ Backup/NZ/GM_IM_45_EH-BN
obs_dirs=/Volumes/SeaJade\ 2\ Backup/NZ/mseed_4-4.5_preferred
echo $obs_dirs
write_dir=/Volumes/SeaJade\ 2\ Backup/NZ/GM_IM_4-4.5_fas_2
# PSA_periods=(0.01 0.015 0.02 0.03 0.04 0.05 0.07 0.1 0.15 0.2 0.3 0.4 0.5 0.7 1.0 1.5 2.0 3.0 4.0 5.0 7.0 10.0)
PSA_periods=(0.01 0.02 0.03 0.04 0.05 0.075 0.1 0.12 0.15 0.17 0.2 0.25 0.3 0.4 0.5 0.6 0.7 0.75 0.8 0.9 1.0 1.25 1.5 2.0 2.5 3.0 4.0 5.0 6.0 7.5 10.0)
# PSA_periods=(0.01 0.015 0.02 0.03 0.04 0.05 0.07 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0 5.5 6.0 6.5 7.0 7.5 8.0 8.5 9.0 9.5 10.0)

# Make sure to reduce the number of asterisks in the line below if adding additional subfolders
# to the obs_dirs variable.

### If including specific event
# for D in ${obs_dirs}
### If including month subdirectories
# for D in ${obs_dirs}/*
### If including year subdirectories
# for D in ${obs_dirs}/*/*
### If not including year subdirectories
for D in ${obs_dirs}/*/*/*
do
    if [[  `find ${D} -name accBB | wc -l` -ge 1 ]]
    then
        im_dir_name=`basename $D`
        echo $im_dir_name
        dir_path=$(echo "$D" | awk -F "/" '{print "/"$6"/"$7}')
        if [ -d $write_dir/$dir_path/$im_dir_name ]
        then
        	echo $write_dir/$dir_path/$im_dir_name exists
        else
        	echo $D/*/*/accBB
#         python3 /Volumes/SeaJade\ 2\ Backup/NZ/Programs/IM_calculation/IM_calculation/scripts/calculate_ims.py $D/*/*/accBB a -o $write_dir/$dir_path/$im_dir_name -t o -c 000 090 ver rotd50 rotd100 -p "${PSA_periods[@]}" -np 7 -i gm_all
        	python3 /Volumes/SeaJade\ 2\ Backup/NZ/Programs/IM_calculation/IM_calculation/scripts/calculate_ims.py $D/*/*/accBB a -o $write_dir/$dir_path/$im_dir_name -t o -c 000 090 ver rotd50 rotd100 -p "${PSA_periods[@]}" -m PGA PGV CAV AI Ds575 Ds595 MMI pSA FAS -np 16 -i gm_all        	
        fi
#         python /Volumes/SeaJade\ 2\ Backup/NZ/Programs/IM_calculation/IM_calculation/scripts/calculate_ims.py $D/*/*/accBB a -o $write_dir/$dir_path/$im_dir_name -t o -c 000 090 ver rotd50 rotd100 -m PGA PGV -np 8 -i gm_all
#         python /Volumes/SeaJade\ 2\ Backup/NZ/Programs/IM_calculation/IM_calculation/scripts/calculate_ims.py $D/*/*/accBB a -o $write_dir/$dir_path/$im_dir_name -t o -c ver -p "${PSA_periods[@]}" -np 8 -i ver
    fi
done