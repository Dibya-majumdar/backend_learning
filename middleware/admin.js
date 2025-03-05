function validate(req,res,next){
    try{
        const token="tokenAdmin";
    if(token=="tokenAdmin"){
      next();
    }
    else{
        throw new Error("you are not admin")
    }
}catch(err){
    res.send("got error "+err);
}
}

module.exports={validate};