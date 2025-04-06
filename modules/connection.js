const mongoose=require("mongoose");
const {studentModal}=require("./students")

const connectionSchema=new mongoose.Schema({
    fromUserId:{
        type:mongoose.Schema.Types.ObjectId,
        required:true,
        ref:studentModal //require studentModel first
       
    },
    toUserId:{
        type:mongoose.Schema.Types.ObjectId,
        required:true,
        ref:studentModal    //require studentModel first
      
    },
    status:{
        type:String
    }
   
}, {timestamps:true});

connectionSchema.index({fromUserId:1,toUserId:1});

const connectionModel=mongoose.model("connection",connectionSchema);
module.exports={connectionModel};





