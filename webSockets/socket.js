const socket=require("socket.io");

const initializeSocket=(server)=>{
    const io=socket(server,{           //we give http server inside socket so now in this http backend server websockets configured and we can control(send,receive data from frontend ..etc )
        cors:{                   //it is cors configuration that which frontend application can access this websocket
            origin:"http://localhost:5173"
        }
    });
    io.on("connection",(socket)=>{        //so io is like the boss but socket here represnt the user who connects to the server.
        //handle events
        socket.on("joinChat",({userId,targetUserId})=>{         //so this event will run when the user call this event joincaht
            const roomId=[userId,targetUserId].sort().join("_");
            socket.join(roomId);
            console.log("room id is:" ,roomId);
         })      
        socket.on("sendMessage",({firstName,userId,targetUserId,text})=>{
            const roomId=[userId,targetUserId].sort().join("_");
            io.to(roomId).emit("messageReceived",{firstName,text})
        });
        socket.on("disconnect",()=>{});
    })
}


module.exports=initializeSocket;