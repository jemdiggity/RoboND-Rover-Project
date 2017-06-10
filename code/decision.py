import numpy as np
import time

def steer(Rover, angles, bias=0):
    Rover.steer_array = np.roll(Rover.steer_array,1)
    Rover.steer_array[0] = np.clip(np.median(angles * 180/np.pi) +
        bias*np.std(angles * 180/np.pi), -30, 30)
    return np.dot(Rover.steer_array, Rover.steer_filter)

def steer_clear(Rover, value=0):
    steer_array = np.full_like(Rover.steer_array, value)
    
def stuck(Rover):
    return np.allclose(Rover.pos, Rover.last_pos, atol=0.2) and (time.time() - Rover.last_time) > 5

def output(Rover, throttle, brake, steer, clear=False, clear_val=0):
    Rover.throttle = throttle
    Rover.brake = brake
    Rover.steer = steer
    if clear:
        steer_clear(Rover, clear_val)


# This is where you can build a decision tree for determining throttle, brake and steer 
# commands based on the output of the perception_step() function
def decision_step(Rover):

    # Implement conditionals to decide what to do given perception data
    # Here you're all set up with some basic functionality but you'll need to
    # improve on this decision tree to do a good job of navigating autonomously!

    #initialize
    if Rover.last_pos == None:
        Rover.last_pos = np.copy(Rover.pos)
    if Rover.last_time == None:
        Rover.last_time = time.time()
    
    #state machine
    if Rover.state == 'picking up':
        output(Rover,0,0,0, True)
        if not Rover.picking_up:
            Rover.state = 'forward'
    elif Rover.state == 'stuck!':
        output(Rover, -1, 0, 0, True)
        if time.time() - Rover.stuck_at > 2:
            Rover.state = 'stop'
    elif Rover.state == 'forward':
        if not np.allclose(Rover.pos, Rover.last_pos, atol=0.2):
            Rover.last_pos = np.copy(Rover.pos)
            Rover.last_time = time.time()
            
        if Rover.picking_up:
            Rover.state = 'picking up'
            output(Rover, 0, 0, 0, True)
        elif stuck(Rover):
            Rover.state = 'stuck!'
            Rover.stuck_at = time.time()
            output(Rover, 0, 0, 0, True)
        elif Rover.nav_angles is not None:
            if Rover.near_sample or len(Rover.rock_dists):
                output(Rover, 0, Rover.brake_set, 0, True)
                Rover.state = 'stop before rock'
            elif len(Rover.nav_angles) >= Rover.stop_forward:  
                output(Rover, Rover.throttle_set if Rover.vel < Rover.max_vel else 0, 0, steer(Rover, Rover.nav_angles, 0.1))
            elif len(Rover.nav_angles) < Rover.stop_forward:
                output(Rover, 0, 1, 0, True)
                Rover.state = 'stop'
    elif Rover.state == 'stop before rock':
        if Rover.vel <= 0.2:
            Rover.state = 'seek rock'
            output(Rover, 0, 0, 0, True)
        elif Rover.vel > 0.2:
            output(Rover, 0, Rover.brake_set, 0, True)
    elif Rover.state == 'seek rock':
        if not np.allclose(Rover.pos, Rover.last_pos, atol=0.2):
            Rover.last_pos = np.copy(Rover.pos)
            Rover.last_time = time.time()

        if Rover.near_sample:
            output(Rover, 0, Rover.brake_set, 0, True)
            Rover.state = 'stop'
        elif stuck(Rover):
            Rover.state = 'stuck!'
            Rover.stuck_at = time.time()
            output(Rover, 0, 0, 0, True)
        elif len(Rover.rock_angles):
            output(Rover, Rover.throttle_set if Rover.vel < 0.5 * Rover.max_vel else 0, 0, steer(Rover, Rover.rock_angles))
        else:
            Rover.state = 'forward'
            output(Rover, 0, 0, 0)
    elif Rover.state == 'stop':
        if abs(Rover.vel) <= 0.2 and Rover.near_sample:
            Rover.send_pickup = True
            Rover.state = 'picking up'
            output(Rover, 0, Rover.brake_set, 0, True)
        elif abs(Rover.vel) > 0.2:
            output(Rover, 0, Rover.brake_set, 0, True)
        elif abs(Rover.vel) <= 0.1:
            if len(Rover.nav_angles) < Rover.go_forward:
                output(Rover, 0, 0, -15, True, -15)
            elif len(Rover.nav_angles) >= Rover.go_forward:
                output(Rover, Rover.throttle_set, 0, steer(Rover, Rover.nav_angles))
                Rover.state = 'forward'
    else:
        print("empty case")

    return Rover

