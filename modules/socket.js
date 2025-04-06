const mongoose = require('mongoose');
const { studentModal } = require('./students');

const messageSchema=new mongoose.Schema(
    {
        senderId:{
            type:mongoose.Schema.Types.ObjectId,
            ref:studentModal,
            required:true
        },
        text:{
            type:String,
            required:true

        }
    },{timestamps:true}
);

const chatSchema=new mongoose.Schema({
    participants:[
        {type:mongoose.Schema.Types.ObjectId,ref:studentModal,required:true}
    ],
    messages:[messageSchema]
})
const chat=mongoose.model("Chat",chatSchema);
module.exports={chat};