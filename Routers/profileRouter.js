const express=require("express");
const profileRouter=express.Router();
const userAuth=require("../middleware/userAuth");
const { studentModal } = require("../modules/students");
const bcrypt = require('bcrypt');


profileRouter.get("/profile/view", userAuth,  async(req,res)=>{
    try{

   
  const userId=req.user;
    const findUserById=await studentModal.findOne({_id:userId._id});
    if(findUserById==null){
        console.log("pls sign up");
    }
    res.json({
        "messege":"your profile----",
        // findUserById
        userId
    })
    

}catch(err){
    res.json({"error":`${err.messege}`})
}

})
profileRouter.patch("/profile/edit",userAuth,async (req,res)=>{
    try{
        const data=req.user._id;
        console.log(data)
        const {firstName,lastName,age,password,photoUrl}=req.body;
        const gettingEmailFromBody=Object.keys(req.body);
        
        if(gettingEmailFromBody.includes("emailId")){
         
            throw new Error("can't update emailId")

        }
         const finStu= await studentModal.find(data);
           
            if(finStu==null){
                throw new Error("sudent does not exist");
            }

        const passwordHash= await bcrypt.hash(password,10);

        const student= {       //don't use new studentModel .just because of thsi small error i have wasted more thean 3 hours .
            password:passwordHash,
            age,
            firstName,
            lastName,
            photoUrl
        }
        const updataData=await studentModal.findByIdAndUpdate(data,student);
     
        res.json({
            "message":"profile updated successfully!",
            
        })

    }catch(err){
        res.send("error accured  "+err.messege);
    }
})



profileRouter.delete("/profile/delete",userAuth,async(req,res)=>{
    try{
        const data=req.user._id;

        const deleteTable=await studentModal.findByIdAndDelete(data);
        res.json({"message":"deleted successfully"});
    }catch(err){
        res.send('error'+err.message);
    }


})


module.exports={profileRouter};