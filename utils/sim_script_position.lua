function sysCall_init()
    robotHandle=sim.getObjectHandle(sim.handle_self)
    --head joints
    head_x=sim.getObjectHandle("head_x_joint") -- Handle of the right motor
    head_y=sim.getObjectHandle("head_y_joint")
    head_z=sim.getObjectHandle("head_z_joint") 
    
    --right_arm_joints
    right_shoulder_rotate=sim.getObjectHandle("right_shoulder_rotate_joint") 
    right_shoulder_sideways=sim.getObjectHandle("right_shoulder_sideways_joint") 
    right_elbow=sim.getObjectHandle("right_elbow_joint") 
    right_wrist=sim.getObjectHandle("right_wrist_joint") 
    
    --left_arm_joints
    left_shoulder_rotate=sim.getObjectHandle("left_shoulder_rotate_joint") 
    left_shoulder_sideways=sim.getObjectHandle("left_shoulder_sideways_joint") 
    left_elbow=sim.getObjectHandle("left_elbow_joint") 
    left_wrist=sim.getObjectHandle("left_wrist_joint") 
    
    --upper_body_joints
    waist_x=sim.getObjectHandle("waist_x_joint") 
    waist_y=sim.getObjectHandle("waist_y_joint") 
    waist_z=sim.getObjectHandle("waist_z_joint") 
    
    --right_leg_joints
    right_hip_sideways=sim.getObjectHandle("right_hip_sideways_joint") 
    right_hip_forward=sim.getObjectHandle("right_hip_forward_joint") 
    right_thight=sim.getObjectHandle("right_thight_joint") 
    right_knee=sim.getObjectHandle("right_knee_joint") 
    right_upper_ankle=sim.getObjectHandle("right_upper_ankle_joint") 
    right_lower_ankle=sim.getObjectHandle("right_lower_ankle_joint") 
    
    --left_leg_joints
    left_hip_sideways=sim.getObjectHandle("left_hip_sideways_joint") 
    left_hip_forward=sim.getObjectHandle("left_hip_forward_joint") 
    left_thight=sim.getObjectHandle("left_thight_joint") 
    left_knee=sim.getObjectHandle("left_knee_joint") 
    left_upper_ankle=sim.getObjectHandle("left_upper_ankle_joint") 
    left_lower_ankle=sim.getObjectHandle("left_lower_ankle_joint") 
    
    --noseSensor=sim.getObjectHandle("rosInterfaceControlledBubbleRobSensingNose") -- Handle of the proximity sensor
    --drawingCont=sim.addDrawingObject(sim.drawing_linestrip+sim.drawing_cyclic,2,0,-1,200,{1,1,0},nil,nil,{1,1,0})
    -- Launch the ROS client application:
    if simROS then
        sim.addLog(sim.verbosity_scriptinfos,"ROS interface was found.")
        local sysTime=sim.getSystemTimeInMs(-1) 
        local positionTopicName='robPosition'
        local orientationTopicName='robOrientation'
        local visuDataTopicName = 'visu'
        --local leftMotorTopicName='leftMotorSpeed'..sysTime -- we add a random component so that we can have several instances of this robot running
        --local rightMotorTopicName='rightMotorSpeed'..sysTime -- we add a random component so that we can have several instances of this robot running
        --local sensorTopicName='sensorTrigger'..sysTime -- we add a random component so that we can have several instances of this robot running
        local simulationTimeTopicName='simTime'..sysTime -- we add a random component so that we can have several instances of this robot running
        -- Prepare the sensor publisher and the motor speed subscribers:
        --sensorPub=simROS.advertise('/'..sensorTopicName,'std_msgs/Bool')
        
        --simTimePub=simROS.advertise('/'..simulationTimeTopicName,'std_msgs/Float32')
        --leftMotorSub=simROS.subscribe('/'..leftMotorTopicName,'std_msgs/Float32','setLeftMotorVelocity_cb')
        --rightMotorSub=simROS.subscribe('/'..rightMotorTopicName,'std_msgs/Float32','setRightMotorVelocity_cb')
        
        positionPub=simROS.advertise('/'..positionTopicName,'geometry_msgs/Point32')
        orientationPub=simROS.advertise('/'..orientationTopicName,'geometry_msgs/Point32')
        visuPub=simROS.advertise('/'..visuDataTopicName,'std_msgs/Float32MultiArray')
        
        
        simTimePub=simROS.advertise('/'..simulationTimeTopicName,'std_msgs/Float32')
        
        head_x_joint_sub=simROS.subscribe('/head_x_joint','std_msgs/Float32','head_x_joint_cb')
        head_y_joint_sub=simROS.subscribe('/head_y_joint','std_msgs/Float32','head_y_joint_cb')
        head_z_joint_sub=simROS.subscribe('/head_z_joint','std_msgs/Float32','head_z_joint_cb')
        
        right_shoulder_rotate_joint_sub=simROS.subscribe('/right_shoulder_rotate_joint','std_msgs/Float32','right_shoulder_rotate_joint_cb')
        right_shoulder_sideways_joint_sub=simROS.subscribe('/right_shoulder_sideways_joint','std_msgs/Float32','right_shoulder_sideways_joint_cb')
        right_elbow_joint_sub=simROS.subscribe('/right_elbow_joint','std_msgs/Float32','right_elbow_joint_cb')
        right_wrist_joint_sub=simROS.subscribe('/right_wrist_joint','std_msgs/Float32','right_wrist_joint_cb')
        
        left_shoulder_rotate_joint_sub=simROS.subscribe('/left_shoulder_rotate_joint','std_msgs/Float32','left_shoulder_rotate_joint_cb')
        left_shoulder_sideways_joint_sub=simROS.subscribe('/left_shoulder_sideways_joint','std_msgs/Float32','left_shoulder_sideways_joint_cb')
        left_elbow_joint_sub=simROS.subscribe('/left_elbow_joint','std_msgs/Float32','left_elbow_joint_cb')
        left_wrist_joint_sub=simROS.subscribe('/left_wrist_joint','std_msgs/Float32','left_wrist_joint_cb')
        
        waist_x_joint_sub=simROS.subscribe('/waist_x_joint','std_msgs/Float32','waist_x_joint_cb')
        waist_y_joint_sub=simROS.subscribe('/waist_y_joint','std_msgs/Float32','waist_y_joint_cb')
        waist_z_joint_sub=simROS.subscribe('/waist_z_joint','std_msgs/Float32','waist_z_joint_cb')
        
        right_hip_sideways_joint_sub=simROS.subscribe('/right_hip_sideways_joint','std_msgs/Float32','right_hip_sideways_joint_cb')
        right_hip_forward_joint_sub=simROS.subscribe('/right_hip_forward_joint','std_msgs/Float32','right_hip_forward_joint_cb')
        right_thight_joint_sub=simROS.subscribe('/right_thight_joint','std_msgs/Float32','right_thight_joint_cb')
        right_knee_joint_sub=simROS.subscribe('/right_knee_joint','std_msgs/Float32','right_knee_joint_cb')
        right_upper_ankle_joint_sub=simROS.subscribe('/right_upper_ankle_joint','std_msgs/Float32','right_upper_ankle_joint_cb')
        right_lower_ankle_joint_sub=simROS.subscribe('/right_lower_ankle_joint','std_msgs/Float32','right_lower_ankle_joint_cb')
        
        left_hip_sideways_joint_sub=simROS.subscribe('/left_hip_sideways_joint','std_msgs/Float32','left_hip_sideways_joint_cb')
        left_hip_forward_joint_sub=simROS.subscribe('/left_hip_forward_joint','std_msgs/Float32','left_hip_forward_joint_cb')
        left_thight_joint_sub=simROS.subscribe('/left_thight_joint','std_msgs/Float32','left_thight_joint_cb')
        left_knee_joint_sub=simROS.subscribe('/left_knee_joint','std_msgs/Float32','left_knee_joint_cb')
        left_upper_ankle_joint_sub=simROS.subscribe('/left_upper_ankle_joint','std_msgs/Float32','left_upper_ankle_joint_cb')
        left_lower_ankle_joint_sub=simROS.subscribe('/left_lower_ankle_joint','std_msgs/Float32','left_lower_ankle_joint_cb')
        
        
        
        
        -- Now we start the client application:
        result=sim.launchExecutable('ros_hugo',simulationTimeTopicName,0)
    else
        sim.addLog(sim.verbosity_scripterrors,"ROS interface was not found. Cannot run.")
    end
    
end


function sysCall_sensing() 
    --local p=sim.getObjectPosition(robotHandle,-1)
    --sim.addDrawingObjectItem(drawingCont,p)
end 


function head_x_joint_cb(msg)
    sim.setJointTargetPosition(head_x,msg.data);
end


function head_y_joint_cb(msg)
    sim.setJointTargetPosition(head_y,msg.data);
end


function head_z_joint_cb(msg)
    sim.setJointTargetPosition(head_z,msg.data);
end


function right_shoulder_rotate_joint_cb(msg)
    sim.setJointTargetPosition(right_shoulder_rotate,msg.data);
end


function right_shoulder_sideways_joint_cb(msg)
    sim.setJointTargetPosition(right_shoulder_sideways,msg.data);
end


function right_elbow_joint_cb(msg)
    sim.setJointTargetPosition(right_elbow,msg.data);
end


function right_wrist_joint_cb(msg)
    sim.setJointTargetPosition(right_wrist,msg.data);
end


function left_shoulder_rotate_joint_cb(msg)
    sim.setJointTargetPosition(left_shoulder_rotate,msg.data);
end


function left_shoulder_sideways_joint_cb(msg)
    sim.setJointTargetPosition(left_shoulder_sideways,msg.data);
end


function left_elbow_joint_cb(msg)
    sim.setJointTargetPosition(left_elbow,msg.data);
end


function left_wrist_joint_cb(msg)
    sim.setJointTargetPosition(left_wrist,msg.data);
end


function waist_x_joint_cb(msg)
    sim.setJointTargetPosition(waist_x,msg.data);
end


function waist_y_joint_cb(msg)
    sim.setJointTargetPosition(waist_y,msg.data);
end


function waist_z_joint_cb(msg)
    sim.setJointTargetPosition(waist_z,msg.data);
end


function right_hip_sideways_joint_cb(msg)
    sim.setJointTargetPosition(right_hip_sideways,msg.data);
end


function right_hip_forward_joint_cb(msg)
    sim.setJointTargetPosition(right_hip_forward,msg.data);
end


function right_thight_joint_cb(msg)
    sim.setJointTargetPosition(right_thight,msg.data);
end


function right_knee_joint_cb(msg)
    sim.setJointTargetPosition(right_knee,msg.data);
end


function right_upper_ankle_joint_cb(msg)
    sim.setJointTargetPosition(right_upper_ankle,msg.data);
end


function right_lower_ankle_joint_cb(msg)
    sim.setJointTargetPosition(right_lower_ankle,msg.data);
end


function left_hip_sideways_joint_cb(msg)
    sim.setJointTargetPosition(left_hip_sideways,msg.data);
end


function left_hip_forward_joint_cb(msg)
    sim.setJointTargetPosition(left_hip_forward,msg.data);
end


function left_thight_joint_cb(msg)
    sim.setJointTargetPosition(left_thight,msg.data);
end


function left_knee_joint_cb(msg)
    sim.setJointTargetPosition(left_knee,msg.data);
end


function left_upper_ankle_joint_cb(msg)
    sim.setJointTargetPosition(left_upper_ankle,msg.data);
end


function left_lower_ankle_joint_cb(msg)
    sim.setJointTargetPosition(left_lower_ankle,msg.data);
end




function getTransformStamped(objHandle,name,relTo,relToName)
    t=sim.getSystemTime()
    p=sim.getObjectPosition(objHandle,relTo)
    o=sim.getObjectQuaternion(objHandle,relTo)
    return {
        header={
            stamp=t,
            frame_id=relToName
        },
        child_frame_id=name,
        transform={
            translation={x=p[1],y=p[2],z=p[3]},
            rotation={x=o[1],y=o[2],z=o[3],w=o[4]}
        }
    }
end


function sysCall_actuation()
    -- Send an updated sensor and simulation time message, and send the transform of the robot:
    if simROS then
        --local result=sim.readProximitySensor(noseSensor)
        --local detectionTrigger={}
        --detectionTrigger['data']=result>0
        --simROS.publish(sensorPub,detectionTrigger)
        data=sim.getSimulationTime()
        simROS.publish(simTimePub,{data=sim.getSimulationTime()})
        -- Send the robot's transform:
        simROS.sendTransform(getTransformStamped(robotHandle,'rosInterfaceControlledBubbleRob',-1,'world'))
        -- To send several transforms at once, use simROS.sendTransforms instead
        bodySensor=sim.getObjectHandle("middle_respondable")
        worldSensor=sim.getObjectHandle("ResizableFloor_5_25")
        p=sim.getObjectPosition(bodySensor, worldSensor)
        simROS.publish(positionPub,{x=p[1],y=p[2],z=p[3]})
        o=sim.getObjectOrientation(bodySensor, worldSensor)
        simROS.publish(orientationPub,{x=o[1],y=o[2],z=o[3]})
        --simROS.publish(visuPub, {t=data, xp=p[1], yp=p[2], zp=p[3], xo=o[1], yo=o[2], zo=o[3]})
        --local mytable = {data,p[1],p[2],p[3],o[1],o[2],o[3]}
        mytable = {}
        mytable.data = {data,p[1],p[2],p[3],o[1],o[2],o[3]}
        --mytable.t = data
        --mytable.xp = p[1]
        --mytable.yp = p[2]
        --mytable.zp = p[3]
        --mytable.xo = o[1]
        --mytable.yo = o[2]
        --mytable.zo = o[3]
        
        --mytable[1] = data
        --mytable[2] = p[1]
        --mytable[3] = p[2]
        --mytable[4] = p[3]
        --mytable[5] = o[1]
        --mytable[6] = o[2]
        --mytable[7] = o[3]
        
        sim.addLog(sim.verbosity_scriptinfos,type(mytable))
        simROS.publish(visuPub, mytable)
        
    end
end

function sysCall_cleanup()
    if simROS then
        -- Following not really needed in a simulation script (i.e. automatically shut down at simulation end):
        --simROS.shutdownPublisher(sensorPub)
        simROS.shutdownSubscriber(head_x_joint_sub)
        simROS.shutdownSubscriber(head_y_joint_sub)
        simROS.shutdownSubscriber(head_z_joint_sub)
        
        simROS.shutdownSubscriber(right_shoulder_rotate_joint_sub)
        simROS.shutdownSubscriber(right_shoulder_sideways_joint_sub)
        simROS.shutdownSubscriber(right_elbow_joint_sub)
        simROS.shutdownSubscriber(right_wrist_joint_sub)
        
        simROS.shutdownSubscriber(left_shoulder_rotate_joint_sub)
        simROS.shutdownSubscriber(left_shoulder_sideways_joint_sub)
        simROS.shutdownSubscriber(left_elbow_joint_sub)
        simROS.shutdownSubscriber(left_wrist_joint_sub)
        
        simROS.shutdownSubscriber(waist_x_joint_sub)
        simROS.shutdownSubscriber(waist_y_joint_sub)
        simROS.shutdownSubscriber(waist_z_joint_sub)
        
        simROS.shutdownSubscriber(right_hip_sideways_joint_sub)
        simROS.shutdownSubscriber(right_hip_forward_joint_sub)
        simROS.shutdownSubscriber(right_thight_joint_sub)
        simROS.shutdownSubscriber(right_knee_joint_sub)
        simROS.shutdownSubscriber(right_upper_ankle_joint_sub)
        simROS.shutdownSubscriber(right_lower_ankle_joint_sub)
        
        simROS.shutdownSubscriber(left_hip_sideways_joint_sub)
        simROS.shutdownSubscriber(left_hip_forward_joint_sub)
        simROS.shutdownSubscriber(left_thight_joint_sub)
        simROS.shutdownSubscriber(left_knee_joint_sub)
        simROS.shutdownSubscriber(left_upper_ankle_joint_sub)
        simROS.shutdownSubscriber(left_lower_ankle_joint_sub)
    end
end
