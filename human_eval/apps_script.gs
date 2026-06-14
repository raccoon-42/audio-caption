// Google Apps Script web app: receives a submission from the eval site and
// appends one row per pair-response to the bound Google Sheet.
//
// Setup (see human_eval/README.md):
//   1. Create a Google Sheet, Extensions > Apps Script, paste this in.
//   2. Deploy > New deployment > Web app.
//        Execute as: Me.   Who has access: Anyone.
//   3. Copy the Web app URL into human_eval/site/config.js (EVAL_ENDPOINT).

var HEADERS = [
  "received_at", "rater", "pair_id", "q1", "q2",
  "listens", "ms", "started_at", "finished_at", "user_agent",
];

function doPost(e) {
  var lock = LockService.getScriptLock();
  lock.waitLock(30000);
  try {
    var sheet = SpreadsheetApp.getActiveSpreadsheet().getSheets()[0];
    if (sheet.getLastRow() === 0) {
      sheet.appendRow(HEADERS);
    }
    var data = JSON.parse(e.postData.contents);
    var now = new Date();
    var ids = Object.keys(data.responses || {});
    for (var i = 0; i < ids.length; i++) {
      var id = ids[i];
      var r = data.responses[id];
      sheet.appendRow([
        now, data.rater, id, r.q1, r.q2,
        r.listens, r.ms, data.started_at, data.finished_at, data.user_agent,
      ]);
    }
    return ContentService
      .createTextOutput(JSON.stringify({ ok: true, rows: ids.length }))
      .setMimeType(ContentService.MimeType.JSON);
  } finally {
    lock.releaseLock();
  }
}
