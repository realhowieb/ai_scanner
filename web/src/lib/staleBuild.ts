/**
 * After a deploy, a page loaded from the previous build still points at that build's
 * /_next/static files, which the new server no longer has. The stylesheet and scripts
 * 404, so the page shows unstyled and never leaves "Restoring your session…".
 *
 * This inline script (runs before the app) spots that and reloads the page once, which
 * fetches the new build's HTML. A timestamp in sessionStorage stops a reload loop if the
 * files are missing for some other reason.
 */
export const STALE_BUILD_SCRIPT = `(function(){
var KEY="hsf-stale-reload",WAIT=60000;
function ours(u){return typeof u==="string"&&u.indexOf("/_next/static/")!==-1;}
function reload(){
  try{var last=+(sessionStorage.getItem(KEY)||0);if(Date.now()-last<WAIT)return;sessionStorage.setItem(KEY,String(Date.now()));}catch(e){return;}
  location.reload();
}
window.addEventListener("error",function(e){
  var t=e.target;
  if(t&&(t.tagName==="LINK"||t.tagName==="SCRIPT")&&ours(t.href||t.src))reload();
},true);
window.addEventListener("unhandledrejection",function(e){
  var r=e.reason;
  if(r&&(r.name==="ChunkLoadError"||/Loading (CSS )?chunk .* failed/i.test(String(r.message||""))))reload();
});
window.addEventListener("load",function(){
  var links=document.querySelectorAll('link[rel="stylesheet"]');
  for(var i=0;i<links.length;i++){if(ours(links[i].href)&&!links[i].sheet){reload();return;}}
});
})();`;
