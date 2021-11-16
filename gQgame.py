#################################################
#                                               #
#  gQgame - A game for Q Learning Application   #
#                                               #
#  @author MGokcayK                             #
#                                               #
#################################################

# import general libraries
import os
import pygame
import random
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import button


# import specific scripts
from agent import Agent

# numpy print options arrangement
np.set_printoptions(precision=4)

# initialize game window pozition
os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (200,100)

# initialize FPS
FPS = 60

# color lookup 
COLOR_LOOKUP = {
	'BLACK' : (  0,   0,   0),
	'WHITE' : (255, 255, 255),
	'RED'   : (255,   0,   0),
	'GREEN' : (  0, 255,   0),
	'BLUE'  : (  0,   0, 255)
}

# main object (car) class
class gQcar:
	def __init__(self):
		self.img = pygame.image.load('carLast.png')
		self.x = 0 # locations 
		self.y = 0
		self.vX = 0 # velocities
		self.vY = 0
		self.carW = self.img.get_rect()[2] # car width
		self.carH = self.img.get_rect()[3] # car height
		self.SR = 180 # sensor range
		self.sensor_angles= [self.to_radian(0),
							self.to_radian(22.5),
							self.to_radian(45),
							self.to_radian(67.5),
							self.to_radian(90),
							self.to_radian(112.5),
							self.to_radian(135),
							self.to_radian(157.5),
							self.to_radian(180),
							self.to_radian(202.5),
							self.to_radian(225),
							self.to_radian(245.5),
							self.to_radian(270),
							self.to_radian(292.5),
							self.to_radian(315),
							self.to_radian(337.5)]
		self.sensor_range = []
		for i in range(len(self.sensor_angles)):
			self.sensor_range.append([np.sin(self.sensor_angles[i])*self.SR, -np.cos(self.sensor_angles[i])* self.SR])

	
	def to_radian(self,degree):
		return degree * np.pi / 180

# game class
class gQgame:
	# initialize the game
	def __init__(self, intro=True, train=True, test=False):
		# initialize pygame
		pygame.init()
		pygame.font.init()
		## Initialize display properties 
		pygame.display.set_caption('gQgame')
		self.displayWidth = 800
		self.displayHeight = 800
		self.gameDisplay = pygame.display.set_mode((self.displayWidth,self.displayHeight))
		self.gameDisplay.fill(COLOR_LOOKUP['WHITE'])
		self.clock = pygame.time.Clock()
		## initialize game loop condition
		self.terminal = False
		## initialize car properties 
		self.car = gQcar()
		self.car.x = self.displayWidth/2
		self.car.y = self.displayHeight - 300 - self.car.img.get_rect()[3]
		self.carFunc() 
		## initialize game loop and boundaries 
		self.passed_object = 0
		self.reward = 0
		self.score = 0
		self.paused = False
		self.obstacles_will_regen = False
		self.detect_object = False
		self.active_sensor = None
		self.sensor_list = []
		self.sensor_val = np.zeros(len(self.car.sensor_range), dtype=np.float32)
		self.pre_sensor_val = None
		self.obstacle_obj = None
		self.walls = []
		self.epoch = 350
		self.epoch_box_active = False
		self.intro = intro
		self.textbox_active = False
		self.textbox_text = ""
		self.buttonInitialize()
		self.R1 = True
		if not intro:
			self.obstacles()
			self.Network()
			if train:
				self.train()
			if test:
				self.test()
		else:
			
			self.gameIntro()		

	# resetting game to menu screen
	def reset(self):
		self.__init__()

	# initialize buttons of intro and its callback functions
	def buttonInitialize(self):
		buttonFont = pygame.font.SysFont('Comic Sans MS', 20)
		self.buttons = []
		self.buttons.append(button.Button("Model Name : ", buttonFont,
										COLOR_LOOKUP["BLACK"], (50, 300, 145, 50),
										COLOR_LOOKUP["WHITE"], (200, 50, 190), self.modelnameButton))

		self.buttons.append(button.Button("Test Model with R1", buttonFont, 
										COLOR_LOOKUP["BLACK"], (100, 600, 200, 50), 
										COLOR_LOOKUP["BLUE"], (0,0,180), self.testbutton1 ))
									
		self.buttons.append(button.Button("Test Model with R2", buttonFont, 
										COLOR_LOOKUP["BLACK"], (500, 600, 200, 50), 
										COLOR_LOOKUP["BLUE"], (0,0,180), self.testbutton2 ))
										
		self.buttons.append(button.Button("Train Model with R1", buttonFont, 
										COLOR_LOOKUP["BLACK"], (100, 500, 200, 50), 
										COLOR_LOOKUP["GREEN"], (50,200,160), self.trainButtonR1 ))

		self.buttons.append(button.Button("Train Model with R2", buttonFont, 
										COLOR_LOOKUP["BLACK"], (500, 500, 200, 50), 
										COLOR_LOOKUP["GREEN"], (50,200,160), self.trainButtonR2 ))

		self.buttons.append(button.Button("EXIT", buttonFont, 
										COLOR_LOOKUP["BLACK"], (300, 700, 200, 50), 
										COLOR_LOOKUP["RED"], (200,0,0), self.exitFunc ))
				
	def modelnameButton(self):
		root = tk.Tk()
		root.withdraw()

		file_path = filedialog.askopenfilename()
		fName = file_path.split("/")[-1]
		self.textbox_text = fName.split(".")[0]
		
	def testbutton1(self):
		self.check_input_epoch()
		if self.textbox_text == "":
			print("Please select proper model for testing R1!")
			self.exitFunc()
		self.obstacles()
		self.Network(self.textbox_text)
		self.test()

	def testbutton2(self):
		self.check_input_epoch()
		self.R1 = False
		if self.textbox_text == "":
			print("Please select proper model for testing R2!")
			self.exitFunc()
		self.obstacles()
		self.Network(self.textbox_text)
		self.test()			

	def trainButtonR1(self):
		self.check_input_epoch()
		self.obstacles()
		self.Network(self.textbox_text)					
		self.train()
		
	def trainButtonR2(self):
		self.check_input_epoch()
		self.R1 = False
		self.obstacles()
		self.Network(self.textbox_text)					
		self.train()
		
	def check_input_epoch(self):
		try:
			self.epoch = int(self.epoch)
			if self.epoch / 100000.0 > 1:
				self.epoch = 99999
			self.epoch_box_active = False
		except ValueError:
			print("Oops!  That was no integer number.  Try again...")	

	# resetting game for ai aplication which is not return to menu screen
	def resetForLearning(self):
		self.terminal = False
		self.car = gQcar()
		self.car.x = self.displayWidth/2
		self.car.y = self.displayHeight - 300 - self.car.img.get_rect()[3]
		self.carFunc() 
		self.passed_object = 0
		self.reward = 0
		self.pre_sensor_val = None
		self.score = 0
		self.obstacles()
		self.gameStep()

	# write text on screen 
	def texting(self, text, fontSize, locX, locY):
		gamefont = pygame.font.SysFont('Comic Sans MS', fontSize)
		textSurface = gamefont.render(text,True, COLOR_LOOKUP['BLACK'])
		self.gameDisplay.blit(textSurface, [locX, locY])

	# obstacle generator
	def obstacles(self):
		self.obsW = random.randrange(60,120)
		self.obsH = random.randrange(60,120)
		if self.car.x:
			x1 = self.car.x - self.car.SR * 1.5
			if x1 < 1:
				x1 = 1		
			x2 = self.car.x + self.car.SR * 1.5
			if x2 > self.displayWidth:
				x2 = self.displayWidth - (self.obsW + 1)
			self.obsX = random.randrange(x1,x2)
		else:			
			self.obsX = random.randrange(1, )
		self.obsY = -200
		self.obsS = 4
		self.obsC = random.choice(list(COLOR_LOOKUP.items()))[1]
		# if obstacle color is white, it will change to black to display on white background
		if self.obsC == (255,255,255):
			self.obsC = COLOR_LOOKUP['BLACK']
		self.obstacles_will_regen = False

	# drawing obstacle on screen
	def obstaclesDraw(self):
		self.obstacle_obj = pygame.draw.rect(self.gameDisplay, self.obsC, [self.obsX, self.obsY, self.obsW, self.obsH] )
		self.obsY += self.obsS

	# checking boundaries whether car across them or not
	def boundaries(self):
		line1 = pygame.draw.line(self.gameDisplay, COLOR_LOOKUP['BLACK'], [0, self.displayHeight - 198],
			[self.displayWidth, self.displayHeight - 198], 4 )
		line2 = pygame.draw.line(self.gameDisplay, COLOR_LOOKUP['BLACK'], [0, 0],
			[0, self.displayHeight - 198], 4 )
		line3 = pygame.draw.line(self.gameDisplay, COLOR_LOOKUP['BLACK'], [0, 0],
			[self.displayWidth, 0], 4 )
		line4 = pygame.draw.line(self.gameDisplay, COLOR_LOOKUP['BLACK'], [self.displayWidth-2, 0],
			[self.displayWidth-2, self.displayHeight - 198], 4 )

		if (len(self.walls)==0):
			self.walls.append(line1)
			self.walls.append(line2)
			self.walls.append(line3)
			self.walls.append(line4)
		else:
			self.walls[0] = line1
			self.walls[1] = line2
			self.walls[2] = line3
			self.walls[3] = line4
			
		if self.car.x < 0 or self.car.x > self.displayWidth - self.car.carW:
			self.after_collision()

		if self.car.y < 1 or self.car.y > self.displayWidth - self.car.carH - 200:
			self.after_collision()
			

	def after_collision(self):
		self.texting(' OOOPSS! CRASHED ', 40, 200, 610 )
		self.terminal = True
		if self.R1:
			self.reward -= 1
		else:
			self.reward -= (100 - self.passed_object) / 100 
		self.texting('Passed Object : '+str(self.passed_object), 20, 10, 610 )
		self.obstacles()

	# update car properties and draw on display screen
	def carFunc(self):
		self.car.x += self.car.vX
		self.car.y += self.car.vY
		self.gameDisplay.blit(self.car.img, (self.car.x , self.car.y ))

	# ray_casting is need for understand sensor and objects collision
	def ray_casting(self, point, angle, range, object):
		x_org, y_org = point[0], point[1]
		x, y = x_org, y_org
		distance = 0.0
		while distance < range:
			distance = np.sqrt((x-x_org)**2 + (y-y_org)**2)
			x = x + np.sin(angle) 
			y = y - np.cos(angle)
			if object.collidepoint(x, y):
				return distance, x, y
		return 0, x_org, y_org

	# displaying sensors; if sensor is active its color is red else green
	def sensors(self):
		for i in range(len(self.car.sensor_range)):
			car_x_s = self.car.x + self.car.carW/2
			car_y_s = self.car.y + self.car.carH/2
			car_x_e = car_x_s + self.car.sensor_range[i][0]
			car_y_e = car_y_s + self.car.sensor_range[i][1]
			line = None
			if self.active_sensor is not None:
				if i == self.active_sensor:
					line = pygame.draw.line(self.gameDisplay, COLOR_LOOKUP['RED'], 
						[car_x_s, car_y_s],
						[car_x_e, car_y_e], 
						1 )
					self.active_sensor = None
			else:
				line = pygame.draw.line(self.gameDisplay, COLOR_LOOKUP['GREEN'], 
					[car_x_s, car_y_s],
					[car_x_e, car_y_e], 
					1 )
			if (len(self.sensor_list)<len(self.car.sensor_range)):
				self.sensor_list.append(line)
			else:
				self.sensor_list[i] = line	

	# calculate sensor values
	def sensors_values(self):
		self.sensor_val = np.zeros(len(self.car.sensor_range), dtype=np.float32)

		# calculate sensor values
		def s_inside(i, obj):
			self.active_sensor = i

			distance, x, y = self.ray_casting([self.car.x + self.car.carW/2, self.car.y + self.car.carH/2],
											self.car.sensor_angles[i],
											self.car.SR,
											obj)
			x, y = int(x), int(y)

			pygame.draw.circle(self.gameDisplay, COLOR_LOOKUP["BLUE"], 
							[x,y],
							5)

			self.sensor_val[i] =np.clip(1 - (distance + np.random.normal())/ self.car.SR, 0, 1)
			
			self.sensors()

		sensor_cntr = 0
		for i in range(len(self.sensor_list)):
			for wall in self.walls:
				if wall.colliderect(self.sensor_list[i]):
					s_inside(i, wall)
					sensor_cntr +=1

			if self.obstacle_obj.colliderect(self.sensor_list[i]):
				s_inside(i, self.obstacle_obj)
				sensor_cntr +=1
				self.detect_object = True			
		
		if sensor_cntr != 0:
			if self.R1:
				self.reward -= 0.1 / sensor_cntr #
			else:
				self.reward -= 0.1 * np.sqrt(np.sum(self.sensor_val))  / sensor_cntr #
			
		# randomize sensor error
		rand = np.random.random_sample()
		if rand < 0.15:
			self.sensor_val += np.random.normal(scale=0.01, size=len(self.car.sensor_range))
			self.sensor_val = np.clip(self.sensor_val, 0, 1)
	
		self.texting("Sensors : "+ str(np.round(self.sensor_val[:int(len(self.car.sensor_angles)/2)], 4)), 
					16, 10, 640)
		self.texting("             "+ str(np.round(self.sensor_val[int(len(self.car.sensor_angles)/2):], 4)), 
					16, 10, 660)
		
	# intro menu of game
	def gameIntro(self):
		while self.intro:
			# menu texts
			self.gameDisplay.fill(COLOR_LOOKUP['WHITE'])
			self.texting(' WELCOME TO gQgame! ', 60, 40, 100)
			self.texting(' This game created by MGokcayK for AI applications. ', 20, 140, 220)

			# buttons 
			mousePos   = pygame.mouse.get_pos()
			mouseClick = pygame.mouse.get_pressed()

			for button in self.buttons:
				button.update(mousePos)

			for button in self.buttons:
				button.draw(self.gameDisplay)

			# textbox			
			self.textbox =  pygame.draw.rect(self.gameDisplay, ( 120, 120, 120), (200,300,500,50))
			if 200 < mousePos[0] < 200 + 500 and 300 < mousePos[1] < 300 + 50:
				if mouseClick[0] == 1:
					self.textbox_active = True

			if self.textbox_active:
				self.textbox =  pygame.draw.rect(self.gameDisplay, ( 170, 170, 170), (200,300,500,50))
			
			# epoch box		
			self.epoch_box =  pygame.draw.rect(self.gameDisplay, ( 120, 120, 120), (380,400,80,50))
			if 380 < mousePos[0] < 380 + 80 and 400 < mousePos[1] < 400 + 50:
				if mouseClick[0] == 1:
					self.epoch_box_active = True

			if self.epoch_box_active:
				self.epoch_box =  pygame.draw.rect(self.gameDisplay, ( 170, 170, 170), (380,400,80,50))

			# event section
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					self.intro = False

				if self.textbox_active:
					if event.type == pygame.KEYDOWN:
						if event.key == pygame.K_BACKSPACE:
							self.textbox_text = self.textbox_text[:-1]
						elif event.key == pygame.K_RETURN:
							self.textbox_active = False
						else:
							self.textbox_text += event.unicode
						if len(self.textbox_text) > 66:
							self.textbox_text = self.textbox_text[:67]

				elif self.epoch_box_active:
					if event.type == pygame.KEYDOWN:
						if event.key == pygame.K_BACKSPACE:
							self.epoch = ""
						elif event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
							try:
								self.epoch = int(self.epoch)
								if self.epoch / 100000.0 > 1:
									self.epoch = 99999
								self.epoch_box_active = False
							except ValueError:
								print("Oops!  That was no integer number.  Try again...")	
						else:
							if (type(self.epoch) == int):
								self.epoch = ""
							self.epoch += event.unicode
						
				

				for button in self.buttons:
					button.get_event(event)

			
			self.texting(self.textbox_text, 15, 210, 314)
			self.texting("Epoch : ", 20, 300, 409)
			self.texting(str(self.epoch), 20, 385, 409)

			pygame.display.update()
			self.clock.tick(FPS)		

		self.exitFunc()

	# main game step
	def gameStep(self, actionkey=None):
		self.reward = 0

		self.gameDisplay.fill(COLOR_LOOKUP['WHITE'])

		# key and button conditions
		for event in pygame.event.get():

			if event.type == pygame.QUIT:
				self.terminal = True
				self.exitFunc()

			if event.type == pygame.KEYDOWN:
				if event.key == pygame.K_r:
					self.reset()
				if event.key == pygame.K_t:
					self.resetForLearning()
				if event.key == pygame.K_q:
					self.terminal = True
					self.exitFunc()

		if actionkey == 0: # throttle 
			self.car.vY = -7
			self.car.vX = 0
			self.obsS += 5 * 0

		elif actionkey == 1: # brake 
				self.car.vY = 3
				self.car.vX = 0
				self.obsS -= 1 * 0

		elif actionkey == 2: # left
			self.car.vX = -10
			self.car.vY = 0

		elif actionkey == 3: # right
			self.car.vX = 10
			self.car.vY = 0

		else: # nothing
			self.car.vX = 0
			self.car.vY = 0
		

		# checking crash 
		if (self.car.y < self.obsY + self.obsH) and (self.car.y + self.car.carH > self.obsY) :
			if (self.car.x > self.obsX and self.car.x < self.obsX + self.obsW) or (self.car.x + self.car.carW > self.obsX and self.car.x + self.car.carW < self.obsX + self.obsW):
				self.after_collision()
				state = np.append(self.sensor_val, self.passed_object / 100)
				return state, np.round(self.reward,6), self.terminal, self.paused

		# obstacles regeneration
		if self.detect_object == True:
			if self.obsY+self.obsH > self.displayHeight - 200:
				if not self.obstacles_will_regen:
					self.passed_object += 1
					if self.R1:
						self.reward += 1
					else:
						self.reward += self.passed_object / 100
					self.detect_object = False
				self.obstacles()
				self.gameDisplay.fill(COLOR_LOOKUP['WHITE'])
			elif self.obsY > self.car.y + self.car.carH:
				if not self.obstacles_will_regen:
					self.passed_object += 1
					if self.R1:
						self.reward += 1
					else:
						self.reward += self.passed_object /100
					self.obstacles_will_regen = True
					self.detect_object = False
		else:
			if self.obsY+self.obsH > self.displayHeight - 200:
				self.obstacles()
				self.gameDisplay.fill(COLOR_LOOKUP['WHITE'])
				self.detect_object = False


		# if obstacle has negative velocity it will changes to 3
		if self.obsS < 0:
			self.obsS = 3
		
		# if reach 100, reward will be maximize and resetting the game
		if self.passed_object % 100 == 0 and self.passed_object > 0:
			self.reward += 100
			self.terminal = True

		if self.R1:
			self.reward += 0.01 
		else:
			self.reward += 0.01 * (100 + self.passed_object) / 100

		# game update
		self.carFunc()
		self.sensors()
		self.boundaries()			
		self.obstaclesDraw()
		self.sensors_values()
		self.texting('Passed Object : '+str(self.passed_object), 20, 10, 610 )
		self.texting('Reward : '+str(np.round(self.reward,6)), 20, 210, 610 )
		self.texting('Epsilon : '+str(np.round(self.agent.epsilon,4)), 20, 410, 610 )
		self.score += np.round(self.reward,6)
		self.texting('Score : '+str(np.round(self.score,6)), 20, 610, 610 )
		self.texting("Action : "+ self.pr_action(actionkey), 16, 10, 690)
		if self.agent:
			self.texting("Action Type : "+ self.agent.action_type, 16, 10, 710)
		self.clock.tick(FPS)
		pygame.display.update()
		state = np.append(self.sensor_val, self.passed_object / 100)
		return state, np.round(self.reward, 6), self.terminal, self.paused

	# exit function of game
	def exitFunc(self):
		pygame.quit()
		if self.intro:
			quit()

	# network properties
	def Network(self, MODEL_NAME="gQgame_model"):
		self.batchSize = 64
		self.numActions = 5
		self.inputShape = len(self.car.sensor_angles)+1
		self.learningRate = 5e-6
		self.epsilon = 1
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.996
		self.gamma = 0.99
		self.max_size = 1000000
		self.model_name = MODEL_NAME
		
	# printing actions
	def pr_action(self, actionkey):
		if actionkey == 0:
			return 'Throttle'

		elif actionkey == 1:
			return 'Brake'
		
		elif actionkey == 2:
			return 'Left'

		elif actionkey == 3:
			return 'Right'	

		else:
			return 'Nothing'

	# train function of model
	def train(self):
		f = open('training_history_'+self.model_name + '.txt','w')
		self.agent = Agent(LR=self.learningRate, gamma=self.gamma,
					n_actions= self.numActions,
					input_shape= self.inputShape,
					batch_size= self.batchSize,
					epsilon=self.epsilon, 
					epsilon_dec= self.epsilon_decay,
					epsilon_min= self.epsilon_min,
					model_name=self.model_name)
		scores = []
		avg_scores = []
		eps_history = []
		pass_obj = []
		avg_pass_objs = []
		for i in range(self.epoch):
			done = False
			self.resetForLearning()
			observation, _, _, _ = self.gameStep()
			score = 0

			while not done:
				action = self.agent.choose_action(observation)
				observation_, reward, done, _ = self.gameStep(action)
				score += reward
				self.agent.remember(observation, action, reward, observation_, done)
				observation = observation_
				self.agent.learn()

			if i % 10 == 0 and i > 0:
				self.agent.save_model()
				
			scores.append(score)
			eps_history.append(self.agent.epsilon)
			pass_obj.append(self.passed_object)

			avg_score = np.mean(scores[max(0, i-100):(i+1)])
			avg_scores.append(avg_score)

			avg_pass_obj = np.mean(pass_obj[max(0, i-100):(i+1)])
			avg_pass_objs.append(avg_pass_obj)

			hist = "Epoch {}, Score {}, Avg_Score {:.3f}, Passed_Obj {:d}, Avg_Pass_Obj {:.3f} Epsilon {:.3f}, Mem_cntr {:d}"\
				.format(i, np.round(score, 4), avg_score, self.passed_object, 
					avg_pass_obj, self.agent.epsilon, self.agent.memory.mem_cntr)
			print(hist)
			f.write(hist + '\n')


			if avg_pass_obj > 99:
				break			

		np.save(self.model_name + '_score', scores)
		np.save(self.model_name + '_avg_score', avg_scores)
		np.save(self.model_name + '_eps', eps_history)
		np.save(self.model_name + '_passed_obj', pass_obj)
		np.save(self.model_name + '_avg_passed_obj', avg_pass_objs)		

		eps = np.arange(0,self.epoch)

		fig_score = plt.figure()
		ax = fig_score.add_subplot(1, 1, 1)
		ax.plot(eps, scores)
		ax.set_title('Score vs Epochs')
		ax.set_ylabel('Score')
		ax.set_xlabel('Epochs')
		ax.grid()
		fig_score.savefig(self.model_name+'_score.png')

		fig_avg_score = plt.figure()
		ax2 = fig_avg_score.add_subplot(1, 1, 1)
		ax2.plot(eps, avg_scores)
		ax2.set_title('Average Score vs Epochs')
		ax2.set_ylabel('Average Score')
		ax2.set_xlabel('Epochs')
		ax2.grid()
		fig_avg_score.savefig(self.model_name+'_avg_score.png')

		fig_pass = plt.figure()
		ax3 = fig_pass.add_subplot(1, 1, 1)
		ax3.plot(eps, pass_obj)
		ax3.set_title('Passed Object vs Epochs')
		ax3.set_ylabel('Passed Object')
		ax3.set_xlabel('Epochs')
		ax3.grid()
		fig_pass.savefig(self.model_name+'_passed_obj.png')

		fig_avg_pass = plt.figure()
		ax4 = fig_avg_pass.add_subplot(1, 1, 1)
		ax4.plot(eps, avg_pass_objs)
		ax4.set_title('Average Passed Object vs Epochs')
		ax4.set_ylabel('Average Passed Object')
		ax4.set_xlabel('Epochs')
		ax4.grid()
		fig_avg_pass.savefig(self.model_name+'_avg_passed_obj.png')

		#plt.show()
		f.close()
		self.exitFunc()

	# test function of model
	def test(self):
		scores = []
		eps_history = []
		self.agent = Agent(LR=self.learningRate, gamma=self.gamma,
					n_actions= self.numActions,
					input_shape= self.inputShape,
					batch_size= self.batchSize,
					epsilon=0.01, 
					epsilon_dec= self.epsilon_decay,
					epsilon_min= self.epsilon_min,
					model_name=self.model_name,
					test=True)
					
		for i in range(self.epoch):
			done = False
			self.resetForLearning()
			observation, _, _, _ = self.gameStep()
			score = 0
			while not done:
				action = self.agent.choose_action(observation)
				observation_, reward, done, _ = self.gameStep(action)
				observation = observation_
				score += reward

			eps_history.append(self.agent.epsilon)
			scores.append(score)

			avg_score = np.mean(scores[max(0,i-100):(i+1)])
			print('Epoch: ', i,'score: ', np.round(score, 4),
				' passed obj %d' % self.passed_object)

if __name__ == '__main__':

	gQgame(intro=True, train=False, test=False)
