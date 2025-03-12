const express = require("express");
const connectionRouter = express.Router();
const { connectionModel } = require("../modules/connection");
const userAuth = require("../middleware/userAuth");
const { studentModal } = require("../modules/students");
const mongoose=require("mongoose");

connectionRouter.post(
  "/request/send/:status/:requestid",
  userAuth,
  async (req, res) => {
    try {
      const data = req.user; //login user
      const fromUserId = data._id; // fromuserid

      const status = req.params.status;
      const toUserId = req.params.requestid;

      if (fromUserId == toUserId) {
        throw new Error("you can't send friend request to yourself");
      }
      const toUserChecking = await studentModal.findOne({ _id: toUserId }); //IT(model.find) RETURNS A OBJECT inside a array(wrapping object in a array) OF THE USER(DOCUMENT).but model.findOne returns teh perfect object.IF NOT FOUND THEN RETURNS NULL.
      if (toUserChecking == null) {
        throw new Error("user does not exist in database");
      }

      const restrictStatusValue = ["interested", "ignored"];
      if (!restrictStatusValue.includes(status)) {
        throw new Error("wrong status");
      }

      const checkDbOfConnection = await connectionModel.findOne({
        $or: [
          { fromUserId: fromUserId, toUserId: toUserId },
          { fromUserId: toUserId, toUserId: fromUserId },
        ],
      });
      if (checkDbOfConnection) {
        throw new Error("data already present");
      }
      const connect = new connectionModel({ fromUserId, toUserId, status });
      await connect.save();
      res.json({
        message: "requsested send or store successfully in database",
        connect,
      });
    } catch (err) {
      res.json({
        message: `${err.message}`,
      });
    }
  }
);


//if requestid is fromuserid(who sent the request).but in copy requestid is as the connection documents unique id.
connectionRouter.post("/request/review/:status/:requestid",userAuth,async(req,res)=>{


try{
    const loginUserId=req.user._id;
    const {requestid,status}=req.params;
    const val=["accepted","rejected"];
    if(!val.includes(status)){
        throw new Error("can accept if someone ignored you");
    }
    const checkingReqId=await  connectionModel.findOne({
        fromUserId:new mongoose.Types.ObjectId(requestid), //this new mongoose.Types.ObjectId is must if the stored data is in objectid type.
        toUserId:new mongoose.Types.ObjectId(loginUserId), //this new mongoose.Types.ObjectId is must if the stored data is in objectid type.
        status:"interested"
    }
    );
   if(checkingReqId==null){
        throw new Error("user is not present in our databse,check the id");
    }
    checkingReqId.status=status;
    
    const acceptData=await checkingReqId.save();
    res.json({
        "message":`request ${status} successfully`,
        acceptData
    })
}catch(err){
    res.json({"message":err.message});
}
});










module.exports = { connectionRouter };
