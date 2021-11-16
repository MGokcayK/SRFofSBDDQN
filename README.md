[![Generic badge](https://img.shields.io/badge/DOI-10.31590/ejosat.1008702-blue.svg)](https://doi.org/10.31590/ejosat.1008702)

# SRFofSBDDQN
Implementation of [Setting Reward Function of Sensor Based DDQN Model (SRFofSBDDQN)](https://doi.org/10.31590/ejosat.1008702) article. 

## Environment 
Environment created with [Pygame](https://www.pygame.org/news) module. It has 5 discrete actions. It has 2 different rewarding. Main purpose of the environment is passing 100 objects/obstacles which are created with random size and positions. 

### __Action Space__
| Ind | Act                |
|-----|--------------------|
| 0   | Throttle           |
| 1   | Brake              |
| 2   | Left               |
| 3   | Right              |
| 4   | Nothing            |

## Model Training and Testing
Model created in [gNet](https://github.com/MGokcayK/gNet) which created in pure Python based on NumPy.

![Caption](ex.gif)

```bibtex
@research article { 
    ejosat1008702, 
    journal = {Avrupa Bilim ve Teknoloji Dergisi}, 
    issn = {}, 
    eissn = {2148-2683}, 
    address = {}, 
    publisher = {Osman SAĞDIÇ}, 
    year = {2021}, 
    volume = {}, 
    pages = {539 - 544}, 
    doi = {10.31590/ejosat.1008702}, 
    title = {Setting Reward Function of Sensor Based DDQN Model}, 
    key = {cite}, 
    author = {Kabataş, Mehmet Gökçay and İlhan Omurca, Sevinç} }
```