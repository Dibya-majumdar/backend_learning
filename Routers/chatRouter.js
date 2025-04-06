const express=require("express");
const {studentModal}=require("../modules/students");
const userAuth = require("../middleware/userAuth");
const { chat } = require("../modules/socket");


const chatRouter=express.Router();

chatRouter.get("/chat/:targetUserId",userAuth,async(req,res)=>{
    const {targetUserId}=req.params;
    const userId=req.user._id;

    try{
        let chatt=await chat.findOne({
            participants:{$all:[userId,targetUserId]}
        })

        if(!chatt){
               chatt=new chat({
                participants:[userId,targetUserId],
                messages:[],
                })
               await chatt.save();
        }
res.json(chatt);
    }catch(err){
        console.log(err);
    }
})




module.exports={chatRouter};