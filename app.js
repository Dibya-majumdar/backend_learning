const { error } = require("console");
const http=require("http");

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
const cors=require("cors");//require cors
const initializeSocket = require("./webSockets/socket");
const { chatRouter } = require("./Routers/chatRouter");

app.use(cors({   //now only origin 5173 can acceess the api 
    origin:"http://localhost:5173",
    methods: ["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
   credentials:true,
   allowedHeaders: ["Content-Type", "Authorization"], 
}))

app.options("*", (req, res) => {
    res.header("Access-Control-Allow-Origin", "http://localhost:5173");
    res.header("Access-Control-Allow-Methods", "GET, POST, PATCH, DELETE, OPTIONS");
    res.header("Access-Control-Allow-Headers", "Content-Type, Authorization");
    res.header("Access-Control-Allow-Credentials", "true");
    res.sendStatus(204); // Respond with no content for preflight
});



 app.use(cookieParser());

 app.use("/",authRouter);
app.use("/",profileRouter);
app.use("/",connectionRouter);
app.use("/",userRouter);
app.use("/",chatRouter);

const server=http.createServer(app);
initializeSocket(server);


async function xyz() {
    try {
        await connectDB();
        server.listen(3000, () => {
            console.log("Listening on port 3000");
        });
    } catch (err) {
        console.error("Error starting the server:", err);
    }
}

//middlewares

xyz();

// app.use("/",authRouter);
// app.use("/",profileRouter);
// app.use("/",connectionRouter);
// app.use("/",userRouter);

