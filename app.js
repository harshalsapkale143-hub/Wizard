const candidates=[
["RELIANCE","2,612.40","94","1.8×","Pass","Pass","Breakout","98"],
["TRENT","6,184.20","92","2.1×","Pass","Pass","Breakout","96"],
["BHARTIARTL","1,742.60","89","1.6×","Pass","Pass","Breakout","94"],
["BEL","382.15","87","1.7×","Pass","Pass","Breakout","92"],
["HDFCBANK","1,698.30","84","1.5×","Pass","Pass","Breakout","90"],
["LT","3,091.20","81","1.6×","Pass","Pass","Setup","86"],
["TITAN","3,872.10","79","1.4×","Pass","Pass","Setup","83"]];
const trades=[
["2026-09-01","2026-09-16","RELIANCE","2,456","2,612","+2.3R","+6.4%","15D exit"],
["2026-09-03","2026-09-18","TCS","3,421","3,312","−1.1R","−3.2%","Stop"],
["2026-08-28","2026-09-12","TITAN","3,612","3,872","+1.8R","+7.2%","Trail"],
["2026-08-25","2026-09-09","HDFCBANK","1,643","1,698","+1.0R","+3.4%","15D exit"],
["2026-08-20","2026-09-04","LT","3,284","3,091","−1.5R","−5.9%","Stop"]];
function fillCandidates(){
const e=document.getElementById("candidate-rows");
e.innerHTML=candidates.slice(0,5).map(function(r){return "<tr><td class='symbol'>"+r[0]+"</td><td>₹"+r[1]+"</td><td>"+r[2]+"</td><td>"+r[3]+"</td><td><span class='tag green'>"+r[4]+"</span></td><td><span class='tag green'>"+r[5]+"</span></td><td><span class='tag'>"+r[6]+"</span></td><td><b>"+r[7]+"</b></td></tr>"}).join("")}
function fillScanner(){
const e=document.getElementById("scanner-rows");
e.innerHTML=candidates.map(function(r){return "<tr><td class='symbol'>"+r[0]+"</td><td>₹"+r[1]+"</td><td>"+r[2]+"</td><td>"+((100-Number(r[2]))/10+5|0)+"%</td><td>"+r[3]+"</td><td><span class='tag green'>PASS</span></td><td><span class='tag'>"+r[6]+"</span></td><td>"+r[7]+"/100</td></tr>"}).join("")}
function fillTrades(){
document.getElementById("trade-rows").innerHTML=trades.map(function(r){return "<tr>"+r.map(function(x,i){return "<td class='"+(i===0?"symbol":"")+"'>"+x+"</td>"}).join("")+"</tr>"}).join("")}
function drawLine(id,color,values){
const svg=document.getElementById(id),w=900,h=300,p=18,min=Math.min.apply(null,values),max=Math.max.apply(null,values);
const pts=values.map(function(v,i){return (p+i*(w-2*p)/(values.length-1))+","+(h-p-(v-min)/(max-min)*(h-2*p))}).join(" ");
let html="<g stroke='#183047' stroke-width='1'><line x1='0' x2='"+w+"' y1='"+(h*.25)+"' y2='"+(h*.25)+"'/><line x1='0' x2='"+w+"' y1='"+(h*.5)+"' y2='"+(h*.5)+"'/><line x1='0' x2='"+w+"' y1='"+(h*.75)+"' y2='"+(h*.75)+"'/></g>";
html+="<polyline points='"+pts+"' fill='none' stroke='"+color+"' stroke-width='3' stroke-linecap='round' stroke-linejoin='round'/>";
svg.innerHTML=html}
function drawCharts(){
drawLine("equity-chart","#2d91ff",[10,12,11,16,18,17,22,25,29,28,35,39,44,42,51,57,63,59,70,79,88,93,105,101,112,121,134,128,143,151,164,178,190,204,217,230,249,266,281,300,292,310,315]);
drawLine("dd-chart","#ff6573",[0,-2,-1,-4,-3,-5,-2,-6,-4,-7,-5,-8,-6,-10,-7,-5,-9,-12,-8,-6,-10,-7,-11,-9,-8,-5,-7,-4,-6,-3,-5,-4,-6,-8,-5,-4,-3,-6,-2,-5,-3])}
function heat(){
const h=document.getElementById("heatmap");
for(let i=0;i<72;i++){const d=document.createElement("div"),n=Math.sin(i*2.4)*.7+Math.cos(i*.77)*.3;
if(n>.55)d.className="strong";else if(n>.05)d.className="pos";else if(n<-.55)d.className="bad";else if(n<-.05)d.className="neg";h.appendChild(d)}}
function toast(msg){const t=document.getElementById("toast");t.textContent=msg;t.classList.add("show");setTimeout(function(){t.classList.remove("show")},2200)}
const titles={dashboard:"Backtest Dashboard",scanner:"SEPA Scanner",backtest:"Backtest",fundamentals:"Fundamental Filter",trades:"Trade Log",compare:"Strategy Comparison",reports:"Reports"};
document.querySelectorAll(".nav-item,[data-view]").forEach(function(b){b.addEventListener("click",function(){
const v=b.dataset.view;if(!v)return;
document.querySelectorAll(".nav-item").forEach(function(x){x.classList.toggle("active",x.dataset.view===v)});
document.querySelectorAll(".view").forEach(function(x){x.classList.toggle("active",x.id===v)});
document.getElementById("page-title").textContent=titles[v]||"Wizard"})});
document.getElementById("run").onclick=function(){toast("Backtest queued — connect the research runner to execute live results.")};
document.getElementById("run2").onclick=function(){toast("Backtest configuration saved. Runner integration is the next backend step.")};
document.getElementById("scan").onclick=function(){toast("Latest scan requested.")};
document.getElementById("refresh").onclick=function(){toast("Dashboard refreshed.")};
document.getElementById("search").oninput=function(e){const q=e.target.value.toUpperCase();document.querySelectorAll("#scanner-rows tr").forEach(function(r){r.style.display=r.cells[0].textContent.includes(q)?"":"none"})};
fillCandidates();fillScanner();fillTrades();drawCharts();heat();