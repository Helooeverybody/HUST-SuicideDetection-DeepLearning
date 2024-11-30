const chatInput=document.querySelector(".chat-input textarea");
const sendChatBtn=document.querySelector(".chat-input span");
const chatBox=document.querySelector(".chatbox")
let userMessage;
const generateResponse= (incomingChatLi) =>{
    const messageElement= incomingChatLi.querySelector("p");
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({message: userMessage})
    })
    .then(response => response.json())
    .then(data => {
        // Display the prediction returned by Flask
        messageElement.textContent=data.response;
    })
    .catch(error => {
        messageElement.textContent="Oops!Error!";
    })
    .finally(()=>{
         chatBox.scrollTo(0,chatBox.scrollHeight)});
}
const createChatLi=(message, className)=>{
    const chatLi=document.createElement("li");
    chatLi.classList.add("chat",className);
    let chatContent= className=="outgoing" ? `<p></p>` : `<span class="material-symbols-outlined">smart_toy</span><p></p>`;
    chatLi.innerHTML=chatContent;
    chatLi.querySelector("p").textContent=message;
    return chatLi;
}
const handleChat=() =>{
    userMessage=chatInput.value.trim();
    if (!userMessage) return;
    chatBox.appendChild(createChatLi(userMessage,"outgoing"))
    chatBox.scrollTo(0,chatBox.scrollHeight);
    setTimeout(()=>{
        const incomingChatLi=createChatLi("Thinking...","incoming")
        chatBox.appendChild(incomingChatLi);
        chatBox.scrollTo(0,chatBox.scrollHeight);
        generateResponse(incomingChatLi);
    },1000);
}
sendChatBtn.addEventListener("click",handleChat);