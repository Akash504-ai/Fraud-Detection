// app/static/js/main.js
document.addEventListener('DOMContentLoaded', function(){
  const exampleBtn = document.getElementById('example-btn');
  const textArea = document.getElementById('text');
  if(exampleBtn && textArea){
    exampleBtn.addEventListener('click', ()=>{
      textArea.value = "Breaking: Scientists find new vaccine that cures all viral infections — government hides the information";
    });
  }
});
