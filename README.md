# Hand-Imitation
RL-based learning for a robotic arm to imitate a given hand in a image feed
with handOfJustice as our environment
## The TODO
- [X] feed and simulation of a robotic arm - Yash 
- [X] Basic RL Environment building
- [X] Reward function
- [X] Publish the Environment at openAI - Somnath
- [X] Report Start -Yash 
- [X] Model Training - Saaswath And Atul
- [X] Improve The algos
- [X] Finish Report 

### Sayonara


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