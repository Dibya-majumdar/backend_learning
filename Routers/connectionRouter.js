const express=require("express");
const connectionRouter=express.Router();
const {connectionModel}=require("../modules/connection");
const userAuth=require("../middleware/userAuth");
const { studentModal } = require("../modules/students");

connectionRouter.post("/request/send/:status/:requestid",userAuth,async (req,res)=>{
try{
const data=req.user; //login user 
const fromUserId=data._id;// fromuserid

const status=req.params.status;
const toUserId=req.params.requestid;

if(fromUserId==toUserId){
    throw new Error("you can't send friend request to yourself");
}
const toUserChecking=await studentModal.findOne({_id:toUserId});//IT(model.find) RETURNS A OBJECT inside a array(wrapping object in a array) OF THE USER(DOCUMENT).but model.findOne returns teh perfect object.IF NOT FOUND THEN RETURNS NULL.
if(toUserChecking==null){
    throw new Error("user does not exist in database");
}


const restrictStatusValue=["interested","ignored"];
if(!restrictStatusValue.includes(status)){
throw new Error("wrong status");
}

const checkDbOfConnection = await connectionModel.findOne({
    $or: [
      { fromUserId: fromUserId, toUserId: toUserId },
      { fromUserId: toUserId, toUserId: fromUserId }
    ]
  });

if(checkDbOfConnection){
    throw new Error("data already present");
}




const connect=new connectionModel({fromUserId,toUserId,status});
await connect.save();
res.json({
    "message":"requsested send or store successfully in database",
    connect
})
}catch(err){
    res.json({
        "message":`${err.message}`
    })
}


})







module.exports={connectionRouter};