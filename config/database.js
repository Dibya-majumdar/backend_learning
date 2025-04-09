const mongoose=require('mongoose');
require('dotenv').config();
// console.log(process.env.Database_Connection_string)
const connectDB=async ()=>{
   await mongoose.connect(process.env.Database_Connection_string);
   console.log("DATABASE CONNECTED");
}

module.exports={connectDB};