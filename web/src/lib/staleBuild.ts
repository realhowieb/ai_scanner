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
  if(r&&(r.name==="ChunkLoadError"||/Loading (CSS )?chunk .* failed|Failed to load chunk/i.test(String(r.message||""))))reload();
});
window.addEventListener("load",function(){
  var links=document.querySelectorAll('link[rel="stylesheet"]');
  for(var i=0;i<links.length;i++){if(ours(links[i].href)&&!links[i].sheet){reload();return;}}
});
})();`;

const STALE_CHUNK = /ChunkLoadError|Loading (CSS )?chunk .* failed|Failed to load chunk/i;

/** True for the errors a page from the previous build throws when it asks for a file the new build no longer has. */
export function isStaleChunkError(message: string): boolean {
  return STALE_CHUNK.test(message);
}

/** Reload once to pick up the new build. Shares the inline script's guard, so there is no reload loop. */
export function reloadForStaleBuild(reload: () => void = () => window.location.reload()): boolean {
  try {
    const KEY = "hsf-stale-reload";
    const last = Number(sessionStorage.getItem(KEY) || 0);
    if (Date.now() - last < 60000) return false;
    sessionStorage.setItem(KEY, String(Date.now()));
  } catch {
    return false;
  }
  reload();
  return true;
}
