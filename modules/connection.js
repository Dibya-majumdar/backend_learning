const mongoose=require("mongoose");

const connectionSchema=new mongoose.Schema({
    fromUserId:{
        type:String,
        required:true,
       
    },
    toUserId:{
        type:String,
        required:true,
      
    },
    status:{
        type:String
    }
   
}, {timestamps:true});

connectionSchema.index({fromUserId:1,toUserId:1});

const connectionModel=mongoose.model("connection",connectionSchema);
module.exports={connectionModel};





