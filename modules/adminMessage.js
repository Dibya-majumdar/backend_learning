const mongoose=require("mongoose");
const { studentModal } = require("./students");


const messageSchema = mongoose.Schema({
    text: String,
    sentAt: {
        type: Date,
        default: Date.now
    }
});
const adminMessage=mongoose.Schema({
    emailId:{
        type:String,
        required:true,
        unique:true
       
    },
    Name:{
        type:String,
        required:true
    },
    messages:[messageSchema]
   
},{timestamps:true});

const AdminMessageModel=mongoose.model("adminmessage",adminMessage);

module.exports={AdminMessageModel}