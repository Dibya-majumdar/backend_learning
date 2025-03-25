const express=require("express");
const bcrypt = require('bcrypt');
require("../modules/students")
const {studentModal}=require("../modules/students");

const jwt=require("jsonwebtoken");
const userAuth = require("../middleware/userAuth");



const authRouter=express.Router();

authRouter.post("/signup",async (req,res)=>{
    try{ 
     const {emailId,password,firstName,lastName,age,gender,photoUrl,about}=req.body;
     const keys=Object.keys(req.body);
     if (!keys.includes("emailId") || !keys.includes("password") || !keys.includes("firstName")) { 
        throw new Error("Fill the credentials");
      }
     if(emailId==null && password==null && firstName==null){
         throw new Error("All fields are required");
     }

     const findEmail=await studentModal.findOne({emailId:emailId});
    //  console.log(findEmail);
      if(findEmail != null){
        throw new Error("pls login your gmail is laready registered")
    }

     //  console.log(data)
     const passwordHash= await bcrypt.hash(password,10);
 
     // console.log(passwordHash);
     const student=new studentModal({
         emailId,
         password:passwordHash,
         firstName,
         lastName,
         age,
         gender,
         photoUrl,
         about
 
         
     });
     await student.save();
     
 
     res.json("user added");
    }catch(err){
     res.status(400).json(err.message)
    }
 
 });



 authRouter.post("/login",async (req,res)=>{
     try{
         const {emailId,password}=req.body;
         const keys=Object.keys(req.body);
        
         if(!keys.includes("emailId") || !keys.includes("password")){
            throw new Error("enter pass and email")
         }
       
         if(emailId==null && password==null){
             throw new Error("pls enter email and password");
 
         }
         const ispresent=await studentModal.findOne({emailId:emailId});
        
         if(!ispresent){
             throw new Error("pls signup first");
         }
         const checkPass=await bcrypt.compare(password,ispresent.password);//it returns true or false
         if(!checkPass){
             throw new Error("invalid password")
         }
 //write the logic of token here 
 const token1=await jwt.sign({_id:ispresent.id},"passOfDibya")
 
        //  const token="fukentokenlife";
         res.cookie("token",token1);   //cookie should be sent at athe time of login not inteh time of signup .Remember.
 
         res.json({
            message:"login successful",
            data:ispresent
         });
             
         
     }catch(err){
         res.status(400).json(err.message);
     }
    
 
 
 });

 authRouter.post("/logout",userAuth,(req,res)=>{

    res.cookie("token",null,{ expires: new Date(Date.now()) })
    res.json({"message":"logout successfull"});
 })


 module.exports={authRouter};
