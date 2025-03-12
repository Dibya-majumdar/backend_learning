const express=require("express");
const userAuth = require("../middleware/userAuth");
const { connectionModel } = require("../modules/connection");
const userRouter=express.Router();
const mongoose=require("mongoose")


//if i want to see who send me requests then? so now api for checking connection requests sent by other users to me 
userRouter.get("/user/request",userAuth,async(req,res)=>{

    try{
        const loginUserId=req.user._id;
        const checkingData=await connectionModel.find({
            toUserId:new mongoose.Types.ObjectId(loginUserId),
            status:"interested"
        });
        if(checkingData.length==0){
            res.json({"message":"No request comes!"})
        }
            res.json({
            "message":"your conncetion requests",
            checkingData
         })
    }catch(err){
            res.json({"message":err.message})
    }
})

module.exports={userRouter};