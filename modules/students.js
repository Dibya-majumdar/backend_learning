const mongoose=require("mongoose");

const studentSchema=new mongoose.Schema({
 firstName:{
    type:String,
    required:true
 },
 lastName:{
    type:String
 },
 emailId:{
    type:String,
   //  required:true,
    unique:true,
    trim:true
 },
 password:{
    type:String,
   //  required:true
 },
 photoUrl:{
type:String,
trim:true
 },
 age:{
    type:Number
 },
 gender:{
    type:String
 }
},{timestamps:true});

//code for mongoosemodel
const studentModal=mongoose.model("Student",studentSchema);

module.exports={studentModal};