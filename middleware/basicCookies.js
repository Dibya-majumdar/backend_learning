// const express=require("express");
// const app=express();

const { studentModal } = require("../modules/students");
const jwt=require("jsonwebtoken");


// const cookie=require("cookie-parser");
// app.use(cookieParser());
async function cookie (req,res,next){
    try{
        const cookies=req.cookies.token;
        // console.log(cookies);

        const decodeobj=await jwt.verify(cookies,"passOfDibya");         //it checks the token ,if the password is presnt in the token or not and if present then return a object ,in which objwct the hidden data set by us at time of creation the token(id) will present
        console.log(decodeobj);
        const {_id}=decodeobj;
        const user=studentModal.findById(_id);
        if(!user){
            throw new Error("user not found");
        }


        // if(cookies!="fukentokenlife"){
        // throw new Error("pls login first");
        // }

        next();
    }catch(err){
        res.send("pls login first"+err);
    }
}

module.exports=cookie;

