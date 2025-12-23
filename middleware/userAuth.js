// const express=require("express");
// const app=express();

const { studentModal } = require("../modules/students");
const jwt=require("jsonwebtoken");


// const cookie=require("cookie-parser");
// app.use(cookieParser());
async function userAuth (req,res,next){
    try{
         
console.log("Token:", req.cookies?.token);
        const cookies=req.cookies.token;
        // console.log(cookies);

        const decodeobj=await jwt.verify(cookies,process.env.Jwt_password);         //it checks the token ,if the password is presnt in the token or not and if present then return a object ,in which objwct the hidden data set by us at time of creation the token(id) will present
        // console.log(decodeobj);
        const {_id}=decodeobj;
      
        const user=await studentModal.findOne({_id});
       
        if(!user){
            throw new Error("user not found");
        }
         req.user=user; 
    
        next();
    }catch(err){
        res.status(400).send("pls login first "+ err.message);
    }
}

module.exports=userAuth;

