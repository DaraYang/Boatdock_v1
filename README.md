# Boatdock_v1
This is a gym environment for the Boatdock game. The task requires an agent to choose one boat and sail it successfully from the start to the goal with limited time and gas resources. 
![task overview](img/boat_task.png)

## Example usage
To run algorithms, the main files are located under algos/.

you can specify which boat to by adding arguments:

boatcong: to use the congruent boat
boatincong0: to use the incongruent boat

```
# to run QPAMDP
cd algos/QPAMDP
python run_boat_qpamdp_copy.py --env_name=boatcong

# to run PDQN
cd algos/QPAMDP
python run_boat_pdqn.py --env_name=boatincong0

# to run HPPO
cd algos/HyAR
python main_boat_hppo.py --env=boatcong

# to run HyAR
cd algos/HyAR
python main_embeding_boat_td3.py --env=boatincong0

# to run DLPA
cd algos/DLPA
python main --env=boatcong
```


