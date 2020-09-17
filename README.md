Installation steps:
1. Installing cudd-3.0.0
```
$ git clone https://github.com/ivmai/cudd.git
$ cd cudd
$ ./configure CC=clang CXX=clang++ --enable-silent-rules --enable-shared --enable-obj
$ make -j4 check
(isntall sudo apt install clang -- if getting clang related error)
```

2. Update the following lines in _shield_synthesis/Makefile_

OLD:
```
CUDD_PATH = /usr/local
LIBS       = -L$(CUDD_PATH)/cudd/.libs -l cudd
INCLUDE    = -I$(CUDD_PATH)/cudd
```
NEW:
```
CUDD_PATH = /path/to/newly/installed/cudd
LIBS       = -L$(CUDD_PATH)/cudd/.libs -l cudd
INCLUDE    = -I$(CUDD_PATH)/cudd
```
3. Now **make** _shield_synthesis/Makefile_:
```
$ cd safe-rl-shielding/shield_synthesis
$ make
```
You should see the last output lines if successfully done!!
```
g++ -o shield_synthesizer Dfa.o ShieldMonitor.o ProductDfa.o Synthesizer.o PythonFormatter.o main.o DfaParser.o -L/home/shrikant/Dialogue_Implement/cudd-3.0.0/cudd/.libs -l cudd
chmod u+x shield_synthesizer
```

To make sure everything works as intended:

* Create and activate an environment using virtualenv.

```bash
virtualenv shielded-learning
source shielded-learning/bin/activate
```

* Clone this repository:
```bash
git clone https://github.com/safe-rl/safe-rl-shielding.git
```
* Install the dependencies:
```bash
pip3 install -r requirements.txt
```

* There is a README file in each environment in the env directory that explains how each example can be run.
