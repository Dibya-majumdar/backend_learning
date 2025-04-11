const express=require("express");
const userAuth = require("../middleware/userAuth");
const { AdminMessageModel } = require("../modules/adminMessage");
const bcrypt = require('bcrypt');
const { studentModal } = require("../modules/students");
const jwt=require("jsonwebtoken");

const adminRouter=express.Router();

adminRouter.post("/admin/login",async(req,res)=>{

    try{
        const {emailId,password,adminKey}=req.body;
if(!emailId && !password){
    throw new Error("enter credentials");
}
console.log(adminKey);
console.log(process.env.Admin_Key)
if(adminKey != process.env.Admin_Key){
    throw new Error("Admin verification fails");
}
const ispresent=await studentModal.findOne({emailId:emailId})
console.log(ispresent)
if(!ispresent){
    throw new Error("pls signup first");
}
const pass= await bcrypt.compare(password,ispresent.password);
if(!pass){
    throw new Error("invalid credentials");
}
const token=jwt.sign({_id:ispresent.id},process.env.Jwt_password,{expiresIn:"7d"}) ;
res.cookie("token",token);

res.json({
    message:"login successful",
    data:ispresent
 });
    }catch(err){
        res.status(400).json(err.message);
    }



})


adminRouter.post("/Admin/message",userAuth,async(req,res)=>{
    try{
        const sender=req.user;
        const {messages,emailId,Name}=req.body;
        // console.log(messages,emailId,Name);
        if(!Name){
            throw new Error("Enter your name")
        }
        if(!emailId){
            throw new Error("pls add email address of this login account")
        }
        // console.log(sender.emailId);

        const isEmail=await studentModal.findOne({emailId:sender.emailId});
        if(!isEmail){
            throw new Error("email is not present")
        }
       if(sender.emailId != emailId) {
        throw new Error("invalid credentials pls write the gmail whom through you login")
       }
        const isPresent=await studentModal.findOne({_id:sender._id});
        if(!isPresent){
            throw new Error("user not authenticated")
        }
        if(messages===null || messages.length===0){
            throw new Error("pls enter text")
        }
        const existingMessage = await AdminMessageModel.findOne({ emailId });

        if (existingMessage) {
          // Push to messages array
          existingMessage.Name=Name,
          existingMessage.messages.push({ text: messages });
          await existingMessage.save();   //you have to call .save() after using .push() on a Mongoose document because .push() only updates the in-memory object, not the database.
    
          res.json({ message: "Message added to existing record" });
        } else {
          // Create new document
          const newMessage = new AdminMessageModel({
            emailId,
            Name,
            messages: [{ text: messages }]
          });
    
          await newMessage.save();
          res.json({ message: "New message record created" });
        }
    }catch(err){
        res.status(400).json(err.message);
    }
})

adminRouter.get("/admin/message",userAuth,async(req,res)=>{
    try{
        const messages=await AdminMessageModel.find({});
        res.send(messages);
    }catch(err){
        res.status(400).json(err.message);
    }
})

module.exports={adminRouter};