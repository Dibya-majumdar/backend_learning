// const express=require("express");
// const app=express();


// const cookie=require("cookie-parser");
// app.use(cookieParser());
function cookie(req,res,next){
    try{
        const cookies=req.cookies.token;
        // console.log(cookies);
        if(cookies!="fukentokenlife"){
        throw new Error("pls login first");
        }
        next();
    }catch(err){
        res.send("pls login first"+err);
    }
}

module.exports=cookie;

