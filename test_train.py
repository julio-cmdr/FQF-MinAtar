import sys
sys.path.append(".")
from datetime import datetime
import numpy as np
import os
import sys
from dopamine.discrete_domains import run_experiment
from dopamine.agents.fqf import fqf_agent
from dopamine.agents.dqn import dqn_agent
from dopamine.colab import utils as colab_utils
from absl import flags
import gin.tf

def create_fqf_agent(sess, environment, summary_writer=None):
  """The Runner class will expect a function of this type to create an agent."""
  return fqf_agent.FQFAgent(sess, num_actions=environment.action_space.n)

path = ''
LOG_PATH = os.path.join(path, 'fqf_test')
sys.path.append(path)

gin.parse_config_file('dopamine/agents/fqf/configs/fqf.gin')

fqf_runner = run_experiment.TrainRunner(LOG_PATH, create_fqf_agent)



now = datetime.now() 
print("now =", now)
print('Will train agent, please be patient, may be a while...')

fqf_runner.run_experiment()

print('Done training!')
now = datetime.now()
 
print("now =", now)