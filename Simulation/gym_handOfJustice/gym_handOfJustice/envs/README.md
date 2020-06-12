# Hand_Controller

robo_hand class can control the arm in any way
```python
from hand_controller import robo_hand, getImage
```
## Hand Movements
there are total 12 Dof (12 angles in radians each)
### Finger and Thumb (10)
```python
robo_hand.fold_finger( finger_index, palm_joint_angle, mid_finger_joint_angle )
```
finger index values
0 - thumb  
1 - index  
2 - middle  
3 - ring  
4 - little  
angle limits => 0 to pi/2

### Wrist (1)
```python
robo_hand.move_wrist( wrist_angle )
```
preferable angle limit => -pi/6 to pi/6

### Arm (1)
```python
robo_hand.wave_arm( arm_angle )
```
preferable angle limit => -pi/3 to pi/3

### Array Input
control all joints together with single function
```python
robo_hand.array_input(
    (angle1, angle2), # Thumb angles
    (angle1, angle2), # Index angles
    (angle1, angle2), # Middle angles
    (angle1, angle2), # Ring angles
    (angle1, angle2), # Little angles
    angle,            # Wrist angle
    angle,            # Arm angle
)
```



## Get Hand Image
```python
getImage()
```
returns captured image in RGBA format (Opencv compatible)
