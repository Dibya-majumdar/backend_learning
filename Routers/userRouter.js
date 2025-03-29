const express=require("express");
const userAuth = require("../middleware/userAuth");
const { connectionModel } = require("../modules/connection");
const userRouter=express.Router();
const mongoose=require("mongoose");
const { studentModal } = require("../modules/students");


//if i want to see who send me requests then? so now api for checking connection requests sent by other users to me 
userRouter.get("/user/request",userAuth,async(req,res)=>{

    try{
        const loginUserId=req.user._id;
        const checkingData=await connectionModel.find({
            toUserId:new mongoose.Types.ObjectId(loginUserId),
            status:"interested"
        }).populate("fromUserId",["firstName","lastName","about"," gender","photoUrl","age","-_id"]);  //so bydefault _id comes in mongodb alsways .to remove that we can use "-" before the field we want to exclude
        const onlyfromUserData = checkingData.map((connection) => connection.fromUserId);
        if(checkingData.length===0){
            res.json({"message":"No request comes!"})
        }else{
            res.json({
                "message":"your conncetion requests",
                  "data":onlyfromUserData
               
            })
        }
        
    }catch(err){
            res.json({"message":err.message})
    }
});



//now i want to see all my connections (accepted)
userRouter.get("/user/connections",userAuth,async (req,res)=>{

try{
    const loginUserId=req.user._id; //getting login userId
    const findingData=await connectionModel.find({  //find document(table) from connectionTable databse where status is accepted and toUserid or from useriD IS LOGIN USER ID
        $or:[
            {fromUserId:loginUserId},
            {toUserId:loginUserId}
        ],
        status:"accepted",
    }).populate("fromUserId",["firstName","lastName","age"]).populate("toUserId",["firstName","lastName","age"]);
   
    // console.log(findingData); 
    if(findingData.length==0){
        throw new Error("make connections ...!")
    }
    const data=findingData.map((row)=>{
       
        if(row.fromUserId._id.toString()===loginUserId.toString()){   //if i am not convert id to string then it does not matches the id
            return row.toUserId._id;
        }else{
            return row.fromUserId._id;
        }
       
    });
    // console.log("data",data);

    const checkUserTable=await studentModal.find({_id:data}).select("firstName lastName age about gender photoUrl -_id");
   if(checkUserTable.length==0){
        throw new Error("data not found");
    }

    res.json({
        "message":"here the list of your connections !!",
        checkUserTable
    })
}catch(err){
    res.json({
        "message":err.message
    })
}

    
})



//now we will make api for showing feed(for sending request or ignore)

userRouter.get("/user/feed",userAuth,async(req,res)=>{

    try{

        const loginUserId=req.user._id;
        const page=parseInt(req.query.page)|| 1;  //check the url->localhost:3000/user/feed?page=2&limit=1 .do not give space .just because space i waste so many time to finout my error.
        let limit=parseInt(req.query.limit) ||10;
        const skip=(page-1)*limit;
        limit =limit>50? 50 :limit;

    const feedData=await connectionModel.find({
        $or:[
            {fromUserId:loginUserId},{toUserId:loginUserId}
        ]
    }).select("fromUserId toUserId")
    const hideUsersFromFeed=new Set();
    feedData.forEach((val)=>{
        hideUsersFromFeed.add(val.fromUserId);
        hideUsersFromFeed.add(val.toUserId);
    })

    const feedUsers=await studentModal.find({
        $and:[
            {_id:{$nin:Array.from(hideUsersFromFeed)}},//notIn -> $nin:["val1","val2",....etc] .array must be there
            {_id:{$ne:loginUserId}}//not equal. -> $ne:anyValue
        ]
    }).select("firstName lastName age about photoUrl gender -_id ")
    .skip(skip)
    .limit(limit);
    res.json(feedUsers);


    }catch(err){
        res.json({
            "error":err.message
        })
    }
    





})

module.exports={userRouter};