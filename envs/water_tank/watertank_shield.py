class Shield:

  def __init__(self):
    self.water_level = 0
    self.switch_state = 0

  def tick(self, *args, **kwargs):
    return self.new_tick(*args, **kwargs)

  def new_tick(self, water_level, switch_state, action):
    if switch_state in [2,3]:
        return True
    elif switch_state in [-2,-3]:
        return False
    elif 1<= water_level <=3:
        return True
    elif 94 <= water_level <= 99:
        return False
    else:
        return action

  def new2_tick(self, water_level, switch_state, action):

    if 1 <= water_level <= 3 and switch_state == 1: 
        return True
    elif 97 <= water_level <= 99 and switch_state <= 1:
        return False
    elif switch_state in [2,3]:
        return True
    elif switch_state in [-2,-3]:
        return False
    else:
        return action

  def original_tick(self, water_level, switch_state, action):

    if 1 <= water_level <= 3 and switch_state == 1: 
        return True
    elif 97 <= water_level <= 99 and switch_state == -1:
        return False
    elif switch_state in [2,3]:
        return True
    elif switch_state in [-2,-3]:
        return False
    else:
        return action
