# Hand-Imitation
RL-based learning for a robotic arm to imitate a given hand in a image feed
with [handOfJustice](https://github.com/hex-plex/gym-handOfJustice) as our environment


## Attendance
1. Somnath ([hex-plex](https://github.com/hex-plex))
2. Yash ([numberbee7070](https://github.com/numberbee7070))
3. Saaswath([infini8-13](https://github.com/infini8-13))
4. Atul ([AtuL-KumaR-00](https://github.com/AtuL-KumaR-00))

## To Setup 
``` console
pip install gym-handOfJustice
```
## To Train

we used Actor Critic technique to update the a CNN
examples are RL-train.py and RL-Test.py 
- In RL-train we have built the Actor and the critic model using tensorflow
- In RL-Test we have used stable-baselines SAC model with LnCnnpolicy policy

## Output
These are the best result after training over a limited amount of time

**Note** 
we have used clips from different versions of trained model and environment so there is a edit on these clips that in the gym-handOfJustice==0.0.6 a flip in the environment was added to make feel of the robotic hand more mirror like which can be spotted in the gif files

![Output-1](/normal&four_diff.gif?raw=true)
![Output-2](/3Pose.gif?raw=true)

## The End
Thats all from our side
[Our report](https://docs.google.com/document/d/1_qCllQiJLehKjnqM8FxTcfWmQpp4JSpf9QeZYaxmxv0/edit?usp=sharing)


![Thank-You](/Thank_You.gif?raw=true)