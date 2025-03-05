const mongoose=require('mongoose');

const connectDB=async ()=>{
   await mongoose.connect("mongodb+srv://majumdardibya700:GfoX7QxBL1rBQE6C@cluster0.4cwur.mongodb.net/students");
   console.log("DATABASE CONNECTED");
}

module.exports={connectDB};