const { error } = require("console");
const {validate}=require("./middleware/admin")
const express=require("express");
const app=express();
const {connectDB}=require("./config/database");
const {studentModal}=require("./modules/students");
app.use(express.json());
const mongoose=require("mongoose");
const bcrypt = require('bcrypt');
const cookieParser=require("cookie-parser");
const cookie=require("./middleware/basicCookies")


 app.use(cookieParser());
async function xyz() {
    try {
        await connectDB();
        app.listen(3000, () => {
            console.log("Listening on port 3000");
        });
    } catch (err) {
        console.error("Error starting the server:", err);
    }
}

//middlewares

xyz();





//4 http methods are there get,post,put,patch





app.get("/admin/profile",validate,(req,res)=>{
   
   res.send("admin ,its your profile");

})
app.get("/admin/hader",validate,(req,res)=>{
   res.send("validate the admin");
   

})
app.post("/signup",async (req,res)=>{
   try{ 
    const {emailId,pasword,firstName,lastName,age,gender}=req.body;
    if(emailId==null && pasword==null && firstName==null){
        throw new Error("All fields are required");
    }
    //  console.log(data)
    const passwordHash= await bcrypt.hash(pasword,10);

    // console.log(passwordHash);
    const student=new studentModal({
        emailId,
        pasword:passwordHash,
        firstName,
        lastName,
        age,
        gender

        
    });
    await student.save();
    

    res.send("user added");
   }catch(err){
    res.send("error"+err)
   }

})
app.get("/profile",cookie,async (req,res)=>{
   try{
    
    const email=req.body.emailId;
    const user=await studentModal.find({emailId:email});
    // console.log(user[0].emailId);
    if(user[0].emailId!=email){
        throw new Error("user not found");
    }
    res.send("your data:"+user);
}catch(err){
    res.status(400).send("erro"+err);
}

})
app.delete("/profile",cookie,async (req,res)=>{
try{
    const id=req.body._id;
    if(!id){
        throw new Error("id not found")
    }
const user=await studentModal.findByIdAndDelete(id);
res.send("user deleted successful");
}catch(err){
    res.send("Eroor:"+err);
}

})
app.patch("/profile",cookie,async(req,res)=>{
 try{
    
    const data=req.body._id;
    const email=req.body.emailId;
    const{firstName,lastName,pasword,age,gender}=req.body;
    // console.log(email);
    if(email!=undefined){
        throw new Error("can not update email")
    }
    const finStu= await studentModal.find({_id:data});
   
    if(finStu==null){
        throw new Error("sudent does not exist");
    }
   const passwordHash= await bcrypt.hash(pasword,10);

    const studata={firstName,lastName,age,gender,pasword:passwordHash};
    const user=await studentModal.findByIdAndUpdate({_id:data},studata);
    res.send("updated successfully");
    }catch(err){
        res.send("Error occured "+err);
    }

})

app.post("/login",async (req,res)=>{
    try{
        const {emailId,pasword}=req.body;
        if(emailId==null && pasword==null){
            throw new Error("pls enter email and password");

        }
        const ispresent=await studentModal.findOne({emailId:emailId});
        console.log(ispresent);
        if(!ispresent){
            throw new Error("pls signup first");
        }
        const checkPass=await bcrypt.compare(pasword,ispresent.pasword);//it returns true or false
        if(!checkPass){
            throw new Error("invalid password")
        }
        const token="fukentokenlife";
        res.cookie("token",token);   //cookie should be sent at athe time of login not inteh time of signup .Remember.

        res.send("login sucessfull");
            
        
    }catch(err){
        res.send("Error occured at login "+err);
    }
   


});
