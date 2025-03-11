const { error } = require("console");

const express=require("express");
const app=express();
const {connectDB}=require("./config/database");
const {studentModal}=require("./modules/students");
app.use(express.json());
const mongoose=require("mongoose");

const cookieParser=require("cookie-parser");
const userAuth=require("./middleware/userAuth");
const jwt=require("jsonwebtoken");
const {authRouter}=require("./Routers/authRouter");
const {profileRouter}=require("./Routers/profileRouter")


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






app.use("/",authRouter);
app.use("/",profileRouter);

app.delete("/profile",userAuth,async (req,res)=>{
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
app.patch("/profile",userAuth,async(req,res)=>{
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


