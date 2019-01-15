# Continental_Rifting
A model of continental rifting in Underworld2 using the UWGeodynamics wrapper.

Run a model by doing:

```
python3 -u lmondy_rift.py --total_vel  2 --gap_to_stop_sedi 100   --total_time 30  --res_scale 1.0 --name second
```

where:

`total_vel` is the total extension velocity in cm/yr, 

`gap_to_stop_sedi` is in kilometers. It controls when sedimentation will stop. It looks at the moho marker passive swarm, and if a gap bigger than `gap_to_stop_sedi` occurs, sedimentation will turn off.

`total_time` is the amount of time to run the model for in millions of years,

`res_scale` is a resolution multiplier. Default resolution is about 1 km, so for example a `res_scale` of 0.5 will make it 2 km resolution.

`name` will change the output folder name.
