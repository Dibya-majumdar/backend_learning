const { error } = require("console");

const express=require("express");
const app=express();
const {connectDB}=require("./config/database");
const {studentModal}=require("./modules/students");
app.use(express.json());
const mongoose=require("mongoose");

const cookieParser=require("cookie-parser");
const userAuth=require("./middleware/userAuth");
const jwt=require("jsonwebtoken");
const {authRouter}=require("./Routers/authRouter");
const {profileRouter}=require("./Routers/profileRouter");
const {connectionRouter}=require("./Routers/connectionRouter");
const {userRouter}=require("./Routers/userRouter");


 app.use(cookieParser());
async function xyz() {
    try {
        await connectDB();
        app.listen(3000, () => {
            console.log("Listening on port 3000");
        });
    } catch (err) {
        console.error("Error starting the server:", err);
    }
}

//middlewares

xyz();

app.use("/",authRouter);
app.use("/",profileRouter);
app.use("/",connectionRouter);
app.use("/",userRouter);

