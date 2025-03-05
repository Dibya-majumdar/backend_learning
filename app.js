const { error } = require("console");
const {validate}=require("./middleware/admin")
const express=require("express");
const app=express();
const {connectDB}=require("./config/database");
const {studentModal}=require("./modules/students");
app.use(express.json());
const mongoose=require("mongoose");



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
app.post("/dibya",async (req,res)=>{
    const data=req.body;
    //  console.log(data)
    const student=new studentModal(req.body);
    await student.save();

    res.send("user added")

})
app.get("/dibya",async (req,res)=>{
   try{

    const email=req.body.emailId;
    const user=await studentModal.find({emailId:email});
    console.log(user[0].emailId);
    if(user[0].emailId!=email){
        throw new Error("user not found");
    }
    res.send("your data:"+user);
}catch(err){
    res.status(400).send("erro"+err);
}

})
app.delete("/dibya",async (req,res)=>{
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
app.patch("/dibya",async(req,res)=>{
 try{
    const data=req.body._id;
    const studata=req.body;
    const user=await studentModal.findByIdAndUpdate({_id:data},studata);
    res.send("updated successfully");
    }catch(err){
        res.send("Error occured "+err);
    }

})
