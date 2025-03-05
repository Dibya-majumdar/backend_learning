const mongoose=require("mongoose");

const studentSchema=new mongoose.Schema({
 firstName:{
    type:String
 },
 lastName:{
    type:String
 },
 emailId:{
    type:String
 },
 pasword:{
    type:Number
 },
 age:{
    type:Number
 },
 gender:{
    type:String
 }
});

//code for mongoosemodel
const studentModal=mongoose.model("Student",studentSchema);

module.exports={studentModal};