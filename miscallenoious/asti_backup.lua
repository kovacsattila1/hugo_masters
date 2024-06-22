--lua

sim=require'sim'
simROS = require'simROS'

function sysCall_init() 
    --robotHandle=sim.getObjectHandle(sim.handle_self)
    --asti=sim.getObject(".")

    --get object handles
    rightLegJoint1 = sim.getObjectHandle("./rightLegJoint1")
    rightLegJoint2 = sim.getObjectHandle("./rightLegJoint2")
    rightLegJoint3 = sim.getObjectHandle("./rightLegJoint3")
    rightLegJoint4 = sim.getObjectHandle("./rightLegJoint4")
    rightLegJoint5 = sim.getObjectHandle("./rightLegJoint5")
    rightLegJoint6 = sim.getObjectHandle("./rightLegJoint6")
    
    leftLegJoint1 = sim.getObjectHandle("./leftLegJoint1")
    leftLegJoint2 = sim.getObjectHandle("./leftLegJoint2")
    leftLegJoint3 = sim.getObjectHandle("./leftLegJoint3")
    leftLegJoint4 = sim.getObjectHandle("./leftLegJoint4")
    leftLegJoint5 = sim.getObjectHandle("./leftLegJoint5")
    leftLegJoint6 = sim.getObjectHandle("./leftLegJoint6")
    
    neckJoint1 = sim.getObjectHandle("./neckJoint1")
    neckJoint2 = sim.getObjectHandle("./neckJoint2")
    
    leftArmJoint1 = sim.getObjectHandle("./leftArmJoint1")
    leftArmJoint2 = sim.getObjectHandle("./leftArmJoint2")
    leftArmJoint3 = sim.getObjectHandle("./leftArmJoint3")
    
    rightArmJoint1 = sim.getObjectHandle("./rightArmJoint1")
    rightArmJoint2 = sim.getObjectHandle("./rightArmJoint2")
    rightArmJoint3 = sim.getObjectHandle("./rightArmJoint3")
  
    
    
    if simROS then
        sim.addLog(sim.verbosity_scriptinfos,"ROS interface was found.")
        local sysTime=sim.getSystemTimeInMs(-1) 
        local positionTopicName='robPosition'
        local orientationTopicName='robOrientation'
        local visuDataTopicName = 'visu'
        local simulationTimeTopicName='simTime'..sysTime -- we add a random component so that we can have several instances of this robot running
        local stateTopicName = 'state'

        positionPub = simROS.advertise('/'..positionTopicName,'geometry_msgs/Point32')
        orientationPub = simROS.advertise('/'..orientationTopicName,'geometry_msgs/Point32')
        visuPub = simROS.advertise('/'..visuDataTopicName,'std_msgs/Float32MultiArray')
        simTimePub =simROS.advertise('/'..simulationTimeTopicName,'std_msgs/Float32')
        simTimePub =simROS.advertise('/'..simulationTimeTopicName,'std_msgs/Float32')
        statePub =simROS.advertise('/'..stateTopicName,'std_msgs/Float32MultiArray')
        
        --subscribe to action topic
        action_sub=simROS.subscribe('/action','std_msgs/Float32MultiArray','action_cb')
        
        
        
        -- Now we start the client application:
        result=sim.launchExecutable('ros_asti',simulationTimeTopicName,0)
    else
        sim.addLog(sim.verbosity_scripterrors,"ROS interface was not found. Cannot run.")
    end
    
    


end

function action_cb(msg)
    sim.setJointTargetForce(rightLegJoint1,msg.data[1], true)
    sim.setJointTargetForce(rightLegJoint2,msg.data[2], true)
    sim.setJointTargetForce(rightLegJoint3,msg.data[3], true)
    sim.setJointTargetForce(rightLegJoint4,msg.data[4], true)
    sim.setJointTargetForce(rightLegJoint5,msg.data[5], true)
    sim.setJointTargetForce(rightLegJoint6,msg.data[6], true)
    sim.setJointTargetForce(leftLegJoint1,msg.data[7], true)
    sim.setJointTargetForce(leftLegJoint2,msg.data[8], true)
    sim.setJointTargetForce(leftLegJoint3,msg.data[9], true)
    sim.setJointTargetForce(leftLegJoint4,msg.data[10], true)
    sim.setJointTargetForce(leftLegJoint5,msg.data[11], true)
    sim.setJointTargetForce(leftLegJoint6,msg.data[12], true)
    sim.setJointTargetForce(neckJoint1,msg.data[13], true)
    sim.setJointTargetForce(neckJoint2,msg.data[14], true)
    sim.setJointTargetForce(leftArmJoint1,msg.data[15], true)
    sim.setJointTargetForce(leftArmJoint2,msg.data[16], true)
    sim.setJointTargetForce(leftArmJoint3,msg.data[17], true)
    sim.setJointTargetForce(rightArmJoint1,msg.data[18], true)
    sim.setJointTargetForce(rightArmJoint2,msg.data[19], true)
    sim.setJointTargetForce(rightArmJoint3,msg.data[20], true)

end


function sysCall_cleanup()
    --rsimIK.eraseEnvironment(ikEnv)
end 

function sysCall_actuation() 
    if simROS then
        data=sim.getSimulationTime()
        simROS.publish(simTimePub,{data=sim.getSimulationTime()})
        
        -- Send the robot's transform:
        --simROS.sendTransform(getTransformStamped(robotHandle,'rosInterfaceControlledBubbleRob',-1,'world'))
        
        -- To send several transforms at once, use simROS.sendTransforms instead

        bodySensor = sim.getObjectHandle("/Asti/asti_body")
        worldSensor = sim.getObjectHandle("/ResizableFloor_5_25")
        
        p=sim.getObjectPosition(bodySensor, worldSensor)
        simROS.publish(positionPub,{x=p[1],y=p[2],z=p[3]})
        o=sim.getObjectOrientation(bodySensor, worldSensor)
        simROS.publish(orientationPub,{x=o[1],y=o[2],z=o[3]})
        
        mytable = {}
        mytable.data = {data,p[1],p[2],p[3],o[1],o[2],o[3]}

        --sim.addLog(sim.verbosity_scriptinfos,type(mytable))
        simROS.publish(visuPub, mytable)
        
        --local t=sim.getSimulationTime()*vel % times[#times]
        p1 = sim.getJointPosition(rightLegJoint1)
        p2 = sim.getJointPosition(rightLegJoint2)
        p3 = sim.getJointPosition(rightLegJoint3)
        p4 = sim.getJointPosition(rightLegJoint4)
        p5 = sim.getJointPosition(rightLegJoint5)
        p6 = sim.getJointPosition(rightLegJoint6)
        
        p7 = sim.getJointPosition(leftLegJoint1)
        p8 = sim.getJointPosition(leftLegJoint2)
        p9 = sim.getJointPosition(leftLegJoint3)
        p10 = sim.getJointPosition(leftLegJoint4)
        p11 = sim.getJointPosition(leftLegJoint5)
        p12 = sim.getJointPosition(leftLegJoint6)
        
        p13 = sim.getJointPosition(neckJoint1)
        p14 = sim.getJointPosition(neckJoint2)
        
        p15 = sim.getJointPosition(leftArmJoint1)
        p16 = sim.getJointPosition(leftArmJoint2)
        p17 = sim.getJointPosition(leftArmJoint3)
        
        p18 = sim.getJointPosition(rightArmJoint1)
        p19 = sim.getJointPosition(rightArmJoint2)
        p20 = sim.getJointPosition(rightArmJoint3)
        
        val = {}
        val.data = {p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18,p19,p20}
        -- sim.addLog(sim.verbosity_scriptinfos, tostring(p1))
        -- sim.addLog(sim.verbosity_scriptinfos, tostring(p2))
        -- sim.addLog(sim.verbosity_scriptinfos, tostring(p3))
        -- sim.addLog(sim.verbosity_scriptinfos, tostring(p4))
        -- sim.addLog(sim.verbosity_scriptinfos, tostring(p5))
        -- sim.addLog(sim.verbosity_scriptinfos, tostring(p6))
        -- sim.addLog(sim.verbosity_scriptinfos, tostring(p7))
        -- sim.addLog(sim.verbosity_scriptinfos, tostring(p8))
        -- sim.addLog(sim.verbosity_scriptinfos, tostring(p9))
        -- sim.addLog(sim.verbosity_scriptinfos, tostring(p10))
        -- sim.addLog(sim.verbosity_scriptinfos, tostring(p11))
        -- sim.addLog(sim.verbosity_scriptinfos, tostring(p12))
        -- sim.addLog(sim.verbosity_scriptinfos, tostring(p13))
        -- sim.addLog(sim.verbosity_scriptinfos, tostring(p14))
        -- sim.addLog(sim.verbosity_scriptinfos, tostring(p15))
        -- sim.addLog(sim.verbosity_scriptinfos, tostring(p16))
        -- sim.addLog(sim.verbosity_scriptinfos, tostring(p17))
        -- sim.addLog(sim.verbosity_scriptinfos, tostring(p18))
        -- sim.addLog(sim.verbosity_scriptinfos, tostring(p19))
        -- sim.addLog(sim.verbosity_scriptinfos, tostring(p20))
        -- sim.addLog(sim.verbosity_scriptinfos, '\n')
        --val.data = {1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,}
        
        simROS.publish(statePub, val)
        
    end
end 
