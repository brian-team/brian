


<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN" "http://www.w3.org/TR/html4/loose.dtd">
<html>
<head>
 <link rel="icon" type="image/vnd.microsoft.icon" href="http://www.gstatic.com/codesite/ph/images/phosting.ico">
 
 <script type="text/javascript">
 
 
 
 var codesite_token = null;
 
 
 var logged_in_user_email = null;
 
 
 var relative_base_url = "";
 
 </script>
 
 
 <title>pthread.h - 
 numexpr -
 
 Project Hosting on Google Code</title>
 <meta http-equiv="Content-Type" content="text/html; charset=UTF-8" >
 
 <link type="text/css" rel="stylesheet" href="http://www.gstatic.com/codesite/ph/7642550995449508181/css/ph_core.css">
 
 <link type="text/css" rel="stylesheet" href="http://www.gstatic.com/codesite/ph/7642550995449508181/css/ph_detail.css" >
 
 
 <link type="text/css" rel="stylesheet" href="http://www.gstatic.com/codesite/ph/7642550995449508181/css/d_sb_20080522.css" >
 
 
 
<!--[if IE]>
 <link type="text/css" rel="stylesheet" href="http://www.gstatic.com/codesite/ph/7642550995449508181/css/d_ie.css" >
<![endif]-->
 <style type="text/css">
 .menuIcon.off { background: no-repeat url(http://www.gstatic.com/codesite/ph/images/dropdown_sprite.gif) 0 -42px }
 .menuIcon.on { background: no-repeat url(http://www.gstatic.com/codesite/ph/images/dropdown_sprite.gif) 0 -28px }
 .menuIcon.down { background: no-repeat url(http://www.gstatic.com/codesite/ph/images/dropdown_sprite.gif) 0 0; }
 </style>
</head>
<body class="t4">
 <script type="text/javascript">
 var _gaq = _gaq || [];
 _gaq.push(
 ['siteTracker._setAccount', 'UA-18071-1'],
 ['siteTracker._trackPageview']);
 
 (function() {
 var ga = document.createElement('script'); ga.type = 'text/javascript'; ga.async = true;
 ga.src = ('https:' == document.location.protocol ? 'https://ssl' : 'http://www') + '.google-analytics.com/ga.js';
 (document.getElementsByTagName('head')[0] || document.getElementsByTagName('body')[0]).appendChild(ga);
 })();
 </script>
 <div id="gaia">
 
 <span>
 
 <a href="#" id="projects-dropdown" onclick="return false;"><u>My favorites</u> <small>&#9660;</small></a>
 | <a href="https://www.google.com/accounts/ServiceLogin?service=code&amp;ltmpl=phosting&amp;continue=http%3A%2F%2Fcode.google.com%2Fp%2Fnumexpr%2Fsource%2Fbrowse%2Ftrunk%2Fnumexpr%2Fwin32%2Fpthread.h&amp;followup=http%3A%2F%2Fcode.google.com%2Fp%2Fnumexpr%2Fsource%2Fbrowse%2Ftrunk%2Fnumexpr%2Fwin32%2Fpthread.h" onclick="_CS_click('/gb/ph/signin');"><u>Sign in</u></a>
 
 </span>

 </div>
 <div class="gbh" style="left: 0pt;"></div>
 <div class="gbh" style="right: 0pt;"></div>
 
 
 <div style="height: 1px"></div>
<!--[if IE 6]>
<div style="text-align:center;">
Support browsers that contribute to open source, try <a href="http://www.firefox.com">Firefox</a> or <a href="http://www.google.com/chrome">Google Chrome</a>.
</div>
<![endif]-->

 <div style="font-weight:bold; color:#a03; padding:5px; margin-top:10px; margin-bottom:-10px; text-align:center; background:#ffeac0;">
 Project Hosting will be READ-ONLY <a href='http://www.timeanddate.com/worldclock/fixedtime.html?month=8&day=10&year=2010&hour=7&min=0&sec=0&p1=224'>Tuesday at 7:00am PDT</a> due to brief network maintenance.
 
 </div>




 <table style="padding:0px; margin: 20px 0px 0px 0px; width:100%" cellpadding="0" cellspacing="0">
 <tr style="height: 58px;">
 
 <td style="width: 55px; text-align:center;">
 <a href="/p/numexpr/">
 
 <img src="http://www.gstatic.com/codesite/ph/images/defaultlogo.png" alt="Logo">
 
 </a>
 </td>
 
 <td style="padding-left: 0.5em">
 
 <div id="pname" style="margin: 0px 0px -3px 0px">
 <a href="/p/numexpr/" style="text-decoration:none; color:#000">numexpr</a>
 
 </div>
 <div id="psum">
 <i><a id="project_summary_link" href="/p/numexpr/" style="text-decoration:none; color:#000">Fast numerical array expression evaluator for Python and NumPy.</a></i>
 </div>
 
 </td>
 <td style="white-space:nowrap;text-align:right">
 
 <form action="/hosting/search">
 <input size="30" name="q" value="">
 <input type="submit" name="projectsearch" value="Search projects" >
 </form>
 
 </tr>
 </table>


 
<table id="mt" cellspacing="0" cellpadding="0" width="100%" border="0">
 <tr>
 <th onclick="if (!cancelBubble) _go('/p/numexpr/');">
 <div class="tab inactive">
 <div class="round4"></div>
 <div class="round2"></div>
 <div class="round1"></div>
 <div class="box-inner">
 <a onclick="cancelBubble=true;" href="/p/numexpr/">Project&nbsp;Home</a>
 </div>
 </div>
 </th><td>&nbsp;&nbsp;</td>
 
 
 
 
 <th onclick="if (!cancelBubble) _go('/p/numexpr/downloads/list');">
 <div class="tab inactive">
 <div class="round4"></div>
 <div class="round2"></div>
 <div class="round1"></div>
 <div class="box-inner">
 <a onclick="cancelBubble=true;" href="/p/numexpr/downloads/list">Downloads</a>
 </div>
 </div>
 </th><td>&nbsp;&nbsp;</td>
 
 
 
 
 
 <th onclick="if (!cancelBubble) _go('/p/numexpr/w/list');">
 <div class="tab inactive">
 <div class="round4"></div>
 <div class="round2"></div>
 <div class="round1"></div>
 <div class="box-inner">
 <a onclick="cancelBubble=true;" href="/p/numexpr/w/list">Wiki</a>
 </div>
 </div>
 </th><td>&nbsp;&nbsp;</td>
 
 
 
 
 
 <th onclick="if (!cancelBubble) _go('/p/numexpr/issues/list');">
 <div class="tab inactive">
 <div class="round4"></div>
 <div class="round2"></div>
 <div class="round1"></div>
 <div class="box-inner">
 <a onclick="cancelBubble=true;" href="/p/numexpr/issues/list">Issues</a>
 </div>
 </div>
 </th><td>&nbsp;&nbsp;</td>
 
 
 
 
 
 <th onclick="if (!cancelBubble) _go('/p/numexpr/source/checkout');">
 <div class="tab active">
 <div class="round4"></div>
 <div class="round2"></div>
 <div class="round1"></div>
 <div class="box-inner">
 <a onclick="cancelBubble=true;" href="/p/numexpr/source/checkout">Source</a>
 </div>
 </div>
 </th><td>&nbsp;&nbsp;</td>
 
 
 <td width="100%">&nbsp;</td>
 </tr>
</table>
<table cellspacing="0" cellpadding="0" width="100%" align="center" border="0" class="st">
 <tr>
 
 
 
 
 
 
 <td>
 <div class="st2">
 <div class="isf">
 
 
 
 <span class="inst1"><a href="/p/numexpr/source/checkout">Checkout</a></span> |
 <span class="inst2"><a href="/p/numexpr/source/browse/">Browse</a></span> |
 <span class="inst3"><a href="/p/numexpr/source/list">Changes</a></span> |
 
 <form action="http://www.google.com/codesearch" method="get" style="display:inline"
 onsubmit="document.getElementById('codesearchq').value = document.getElementById('origq').value + ' package:http://numexpr\\.googlecode\\.com'">
 <input type="hidden" name="q" id="codesearchq" value="">
 <input maxlength="2048" size="38" id="origq" name="origq" value="" title="Google Code Search" style="font-size:92%">&nbsp;<input type="submit" value="Search Trunk" name="btnG" style="font-size:92%">
 
 
 
 </form>
 </div>
</div>

 </td>
 
 
 
 <td height="4" align="right" valign="top" class="bevel-right">
 <div class="round4"></div>
 <div class="round2"></div>
 <div class="round1"></div>
 </td>
 </tr>
</table>
<script type="text/javascript">
 var cancelBubble = false;
 function _go(url) { document.location = url; }
</script>


<div id="maincol"
 
>

 
<!-- IE -->




<div class="expand">


<style type="text/css">
 #file_flipper { display: inline; float: right; white-space: nowrap; }
 #file_flipper.hidden { display: none; }
 #file_flipper .pagelink { color: #0000CC; text-decoration: underline; }
 #file_flipper #visiblefiles { padding-left: 0.5em; padding-right: 0.5em; }
</style>
<div id="nav_and_rev" class="heading">
 <div class="round4"></div>
 <div class="round2"></div>
 <div class="round1"></div>
 <div class="box-inner" id="bub">
 <div class="bub-top">
 <div class="pagination" style="margin-left: 2em">
 <table cellpadding="0" cellspacing="0" class="flipper">
 <tbody>
 <tr>
 
 <td>
 <ul class="leftside">
 
 <li><a href="/p/numexpr/source/browse/branches/multithread/numexpr/win32/pthread.h?r=203" title="Previous">&lsaquo;r203</a></li>
 
 </ul>
 </td>
 
 <td><b>r234</b></td>
 
 </tr>
 </tbody>
 </table>
 </div>
 
 <div class="" style="vertical-align: top">
 <div class="src_crumbs src_nav">
 <strong class="src_nav">Source path:&nbsp;</strong>
 <span id="crumb_root">
 
 <a href="/p/numexpr/source/browse/">svn</a>/&nbsp;</span>
 <span id="crumb_links" class="ifClosed"><a href="/p/numexpr/source/browse/trunk/">trunk</a><span class="sp">/&nbsp;</span><a href="/p/numexpr/source/browse/trunk/numexpr/">numexpr</a><span class="sp">/&nbsp;</span><a href="/p/numexpr/source/browse/trunk/numexpr/win32/">win32</a><span class="sp">/&nbsp;</span>pthread.h</span>
 
 
 </div>
 
 </div>
 <div style="clear:both"></div>
 </div>
 </div>
</div>

<style type="text/css">
 
  tr.inline_comment {
 background: #fff;
 vertical-align: top;
 }
 div.draft, div.published {
 padding: .3em;
 border: 1px solid #999; 
 margin-bottom: .1em;
 font-family: arial, sans-serif;
 max-width: 60em;
 }
 div.draft {
 background: #ffa;
 } 
 div.published {
 background: #e5ecf9;
 }
 div.published .body, div.draft .body {
 padding: .5em .1em .1em .1em;
 max-width: 60em;
 white-space: pre-wrap;
 white-space: -moz-pre-wrap;
 white-space: -pre-wrap;
 white-space: -o-pre-wrap;
 word-wrap: break-word;
 }
 div.draft .actions {
 margin-left: 1em;
 font-size: 90%;
 }
 div.draft form {
 padding: .5em .5em .5em 0;
 }
 div.draft textarea, div.published textarea {
 width: 95%;
 height: 10em;
 font-family: arial, sans-serif;
 margin-bottom: .5em;
 }


 
 .nocursor, .nocursor td, .cursor_hidden, .cursor_hidden td {
 background-color: white;
 height: 2px;
 }
 .cursor, .cursor td {
 background-color: darkblue;
 height: 2px;
 display: '';
 }

</style>
<div class="fc">
 
 
 
<style type="text/css">
.undermouse span { 
 background-image: url(http://www.gstatic.com/codesite/ph/images/comments.gif); }
</style>
<table class="opened" id="review_comment_area" 
><tr>
<td id="nums">
<pre><table width="100%"><tr class="nocursor"><td></td></tr></table></pre>

<pre><table width="100%" id="nums_table_0"><tr id="gr_svn204_1"

><td id="1"><a href="#1">1</a></td></tr
><tr id="gr_svn204_2"

><td id="2"><a href="#2">2</a></td></tr
><tr id="gr_svn204_3"

><td id="3"><a href="#3">3</a></td></tr
><tr id="gr_svn204_4"

><td id="4"><a href="#4">4</a></td></tr
><tr id="gr_svn204_5"

><td id="5"><a href="#5">5</a></td></tr
><tr id="gr_svn204_6"

><td id="6"><a href="#6">6</a></td></tr
><tr id="gr_svn204_7"

><td id="7"><a href="#7">7</a></td></tr
><tr id="gr_svn204_8"

><td id="8"><a href="#8">8</a></td></tr
><tr id="gr_svn204_9"

><td id="9"><a href="#9">9</a></td></tr
><tr id="gr_svn204_10"

><td id="10"><a href="#10">10</a></td></tr
><tr id="gr_svn204_11"

><td id="11"><a href="#11">11</a></td></tr
><tr id="gr_svn204_12"

><td id="12"><a href="#12">12</a></td></tr
><tr id="gr_svn204_13"

><td id="13"><a href="#13">13</a></td></tr
><tr id="gr_svn204_14"

><td id="14"><a href="#14">14</a></td></tr
><tr id="gr_svn204_15"

><td id="15"><a href="#15">15</a></td></tr
><tr id="gr_svn204_16"

><td id="16"><a href="#16">16</a></td></tr
><tr id="gr_svn204_17"

><td id="17"><a href="#17">17</a></td></tr
><tr id="gr_svn204_18"

><td id="18"><a href="#18">18</a></td></tr
><tr id="gr_svn204_19"

><td id="19"><a href="#19">19</a></td></tr
><tr id="gr_svn204_20"

><td id="20"><a href="#20">20</a></td></tr
><tr id="gr_svn204_21"

><td id="21"><a href="#21">21</a></td></tr
><tr id="gr_svn204_22"

><td id="22"><a href="#22">22</a></td></tr
><tr id="gr_svn204_23"

><td id="23"><a href="#23">23</a></td></tr
><tr id="gr_svn204_24"

><td id="24"><a href="#24">24</a></td></tr
><tr id="gr_svn204_25"

><td id="25"><a href="#25">25</a></td></tr
><tr id="gr_svn204_26"

><td id="26"><a href="#26">26</a></td></tr
><tr id="gr_svn204_27"

><td id="27"><a href="#27">27</a></td></tr
><tr id="gr_svn204_28"

><td id="28"><a href="#28">28</a></td></tr
><tr id="gr_svn204_29"

><td id="29"><a href="#29">29</a></td></tr
><tr id="gr_svn204_30"

><td id="30"><a href="#30">30</a></td></tr
><tr id="gr_svn204_31"

><td id="31"><a href="#31">31</a></td></tr
><tr id="gr_svn204_32"

><td id="32"><a href="#32">32</a></td></tr
><tr id="gr_svn204_33"

><td id="33"><a href="#33">33</a></td></tr
><tr id="gr_svn204_34"

><td id="34"><a href="#34">34</a></td></tr
><tr id="gr_svn204_35"

><td id="35"><a href="#35">35</a></td></tr
><tr id="gr_svn204_36"

><td id="36"><a href="#36">36</a></td></tr
><tr id="gr_svn204_37"

><td id="37"><a href="#37">37</a></td></tr
><tr id="gr_svn204_38"

><td id="38"><a href="#38">38</a></td></tr
><tr id="gr_svn204_39"

><td id="39"><a href="#39">39</a></td></tr
><tr id="gr_svn204_40"

><td id="40"><a href="#40">40</a></td></tr
><tr id="gr_svn204_41"

><td id="41"><a href="#41">41</a></td></tr
><tr id="gr_svn204_42"

><td id="42"><a href="#42">42</a></td></tr
><tr id="gr_svn204_43"

><td id="43"><a href="#43">43</a></td></tr
><tr id="gr_svn204_44"

><td id="44"><a href="#44">44</a></td></tr
><tr id="gr_svn204_45"

><td id="45"><a href="#45">45</a></td></tr
><tr id="gr_svn204_46"

><td id="46"><a href="#46">46</a></td></tr
><tr id="gr_svn204_47"

><td id="47"><a href="#47">47</a></td></tr
><tr id="gr_svn204_48"

><td id="48"><a href="#48">48</a></td></tr
><tr id="gr_svn204_49"

><td id="49"><a href="#49">49</a></td></tr
><tr id="gr_svn204_50"

><td id="50"><a href="#50">50</a></td></tr
><tr id="gr_svn204_51"

><td id="51"><a href="#51">51</a></td></tr
><tr id="gr_svn204_52"

><td id="52"><a href="#52">52</a></td></tr
><tr id="gr_svn204_53"

><td id="53"><a href="#53">53</a></td></tr
><tr id="gr_svn204_54"

><td id="54"><a href="#54">54</a></td></tr
><tr id="gr_svn204_55"

><td id="55"><a href="#55">55</a></td></tr
><tr id="gr_svn204_56"

><td id="56"><a href="#56">56</a></td></tr
><tr id="gr_svn204_57"

><td id="57"><a href="#57">57</a></td></tr
><tr id="gr_svn204_58"

><td id="58"><a href="#58">58</a></td></tr
><tr id="gr_svn204_59"

><td id="59"><a href="#59">59</a></td></tr
><tr id="gr_svn204_60"

><td id="60"><a href="#60">60</a></td></tr
><tr id="gr_svn204_61"

><td id="61"><a href="#61">61</a></td></tr
><tr id="gr_svn204_62"

><td id="62"><a href="#62">62</a></td></tr
><tr id="gr_svn204_63"

><td id="63"><a href="#63">63</a></td></tr
><tr id="gr_svn204_64"

><td id="64"><a href="#64">64</a></td></tr
><tr id="gr_svn204_65"

><td id="65"><a href="#65">65</a></td></tr
><tr id="gr_svn204_66"

><td id="66"><a href="#66">66</a></td></tr
><tr id="gr_svn204_67"

><td id="67"><a href="#67">67</a></td></tr
><tr id="gr_svn204_68"

><td id="68"><a href="#68">68</a></td></tr
></table></pre>

<pre><table width="100%"><tr class="nocursor"><td></td></tr></table></pre>
</td>
<td id="lines">
<pre class="prettyprint"><table width="100%"><tr class="cursor_stop cursor_hidden"><td></td></tr></table></pre>

<pre class="prettyprint "><table id="src_table_0"><tr
id=sl_svn204_1

><td class="source">/*<br></td></tr
><tr
id=sl_svn204_2

><td class="source"> * Header used to adapt pthread-based POSIX code to Windows API threads.<br></td></tr
><tr
id=sl_svn204_3

><td class="source"> *<br></td></tr
><tr
id=sl_svn204_4

><td class="source"> * Copyright (C) 2009 Andrzej K. Haczewski &lt;ahaczewski@gmail.com&gt;<br></td></tr
><tr
id=sl_svn204_5

><td class="source"> */<br></td></tr
><tr
id=sl_svn204_6

><td class="source"><br></td></tr
><tr
id=sl_svn204_7

><td class="source">#ifndef PTHREAD_H<br></td></tr
><tr
id=sl_svn204_8

><td class="source">#define PTHREAD_H<br></td></tr
><tr
id=sl_svn204_9

><td class="source"><br></td></tr
><tr
id=sl_svn204_10

><td class="source">#ifndef WIN32_LEAN_AND_MEAN<br></td></tr
><tr
id=sl_svn204_11

><td class="source">#define WIN32_LEAN_AND_MEAN<br></td></tr
><tr
id=sl_svn204_12

><td class="source">#endif<br></td></tr
><tr
id=sl_svn204_13

><td class="source"><br></td></tr
><tr
id=sl_svn204_14

><td class="source">#include &lt;windows.h&gt;<br></td></tr
><tr
id=sl_svn204_15

><td class="source"><br></td></tr
><tr
id=sl_svn204_16

><td class="source">/*<br></td></tr
><tr
id=sl_svn204_17

><td class="source"> * Defines that adapt Windows API threads to pthreads API<br></td></tr
><tr
id=sl_svn204_18

><td class="source"> */<br></td></tr
><tr
id=sl_svn204_19

><td class="source">#define pthread_mutex_t CRITICAL_SECTION<br></td></tr
><tr
id=sl_svn204_20

><td class="source"><br></td></tr
><tr
id=sl_svn204_21

><td class="source">#define pthread_mutex_init(a,b) InitializeCriticalSection((a))<br></td></tr
><tr
id=sl_svn204_22

><td class="source">#define pthread_mutex_destroy(a) DeleteCriticalSection((a))<br></td></tr
><tr
id=sl_svn204_23

><td class="source">#define pthread_mutex_lock EnterCriticalSection<br></td></tr
><tr
id=sl_svn204_24

><td class="source">#define pthread_mutex_unlock LeaveCriticalSection<br></td></tr
><tr
id=sl_svn204_25

><td class="source"><br></td></tr
><tr
id=sl_svn204_26

><td class="source">/*<br></td></tr
><tr
id=sl_svn204_27

><td class="source"> * Implement simple condition variable for Windows threads, based on ACE<br></td></tr
><tr
id=sl_svn204_28

><td class="source"> * implementation.<br></td></tr
><tr
id=sl_svn204_29

><td class="source"> *<br></td></tr
><tr
id=sl_svn204_30

><td class="source"> * See original implementation: http://bit.ly/1vkDjo<br></td></tr
><tr
id=sl_svn204_31

><td class="source"> * ACE homepage: http://www.cse.wustl.edu/~schmidt/ACE.html<br></td></tr
><tr
id=sl_svn204_32

><td class="source"> * See also: http://www.cse.wustl.edu/~schmidt/win32-cv-1.html<br></td></tr
><tr
id=sl_svn204_33

><td class="source"> */<br></td></tr
><tr
id=sl_svn204_34

><td class="source">typedef struct {<br></td></tr
><tr
id=sl_svn204_35

><td class="source">	LONG waiters;<br></td></tr
><tr
id=sl_svn204_36

><td class="source">	int was_broadcast;<br></td></tr
><tr
id=sl_svn204_37

><td class="source">	CRITICAL_SECTION waiters_lock;<br></td></tr
><tr
id=sl_svn204_38

><td class="source">	HANDLE sema;<br></td></tr
><tr
id=sl_svn204_39

><td class="source">	HANDLE continue_broadcast;<br></td></tr
><tr
id=sl_svn204_40

><td class="source">} pthread_cond_t;<br></td></tr
><tr
id=sl_svn204_41

><td class="source"><br></td></tr
><tr
id=sl_svn204_42

><td class="source">extern int pthread_cond_init(pthread_cond_t *cond, const void *unused);<br></td></tr
><tr
id=sl_svn204_43

><td class="source">extern int pthread_cond_destroy(pthread_cond_t *cond);<br></td></tr
><tr
id=sl_svn204_44

><td class="source">extern int pthread_cond_wait(pthread_cond_t *cond, CRITICAL_SECTION *mutex);<br></td></tr
><tr
id=sl_svn204_45

><td class="source">extern int pthread_cond_signal(pthread_cond_t *cond);<br></td></tr
><tr
id=sl_svn204_46

><td class="source">extern int pthread_cond_broadcast(pthread_cond_t *cond);<br></td></tr
><tr
id=sl_svn204_47

><td class="source"><br></td></tr
><tr
id=sl_svn204_48

><td class="source">/*<br></td></tr
><tr
id=sl_svn204_49

><td class="source"> * Simple thread creation implementation using pthread API<br></td></tr
><tr
id=sl_svn204_50

><td class="source"> */<br></td></tr
><tr
id=sl_svn204_51

><td class="source">typedef struct {<br></td></tr
><tr
id=sl_svn204_52

><td class="source">	HANDLE handle;<br></td></tr
><tr
id=sl_svn204_53

><td class="source">	void *(*start_routine)(void*);<br></td></tr
><tr
id=sl_svn204_54

><td class="source">	void *arg;<br></td></tr
><tr
id=sl_svn204_55

><td class="source">} pthread_t;<br></td></tr
><tr
id=sl_svn204_56

><td class="source"><br></td></tr
><tr
id=sl_svn204_57

><td class="source">extern int pthread_create(pthread_t *thread, const void *unused,<br></td></tr
><tr
id=sl_svn204_58

><td class="source">			  void *(*start_routine)(void*), void *arg);<br></td></tr
><tr
id=sl_svn204_59

><td class="source"><br></td></tr
><tr
id=sl_svn204_60

><td class="source">/*<br></td></tr
><tr
id=sl_svn204_61

><td class="source"> * To avoid the need of copying a struct, we use small macro wrapper to pass<br></td></tr
><tr
id=sl_svn204_62

><td class="source"> * pointer to win32_pthread_join instead.<br></td></tr
><tr
id=sl_svn204_63

><td class="source"> */<br></td></tr
><tr
id=sl_svn204_64

><td class="source">#define pthread_join(a, b) win32_pthread_join(&amp;(a), (b))<br></td></tr
><tr
id=sl_svn204_65

><td class="source"><br></td></tr
><tr
id=sl_svn204_66

><td class="source">extern int win32_pthread_join(pthread_t *thread, void **value_ptr);<br></td></tr
><tr
id=sl_svn204_67

><td class="source"><br></td></tr
><tr
id=sl_svn204_68

><td class="source">#endif /* PTHREAD_H */<br></td></tr
></table></pre>

<pre class="prettyprint"><table width="100%"><tr class="cursor_stop cursor_hidden"><td></td></tr></table></pre>
</td>
</tr></table>
<script type="text/javascript">
 var lineNumUnderMouse = -1;
 
 function gutterOver(num) {
 gutterOut();
 var newTR = document.getElementById('gr_svn204_' + num);
 if (newTR) {
 newTR.className = 'undermouse';
 }
 lineNumUnderMouse = num;
 }
 function gutterOut() {
 if (lineNumUnderMouse != -1) {
 var oldTR = document.getElementById(
 'gr_svn204_' + lineNumUnderMouse);
 if (oldTR) {
 oldTR.className = '';
 }
 lineNumUnderMouse = -1;
 }
 }
 var numsGenState = {table_base_id: 'nums_table_'};
 var srcGenState = {table_base_id: 'src_table_'};
 var alignerRunning = false;
 var startOver = false;
 function setLineNumberHeights() {
 if (alignerRunning) {
 startOver = true;
 return;
 }
 numsGenState.chunk_id = 0;
 numsGenState.table = document.getElementById('nums_table_0');
 numsGenState.row_num = 0;
 srcGenState.chunk_id = 0;
 srcGenState.table = document.getElementById('src_table_0');
 srcGenState.row_num = 0;
 alignerRunning = true;
 continueToSetLineNumberHeights();
 }
 function rowGenerator(genState) {
 if (genState.row_num < genState.table.rows.length) {
 var currentRow = genState.table.rows[genState.row_num];
 genState.row_num++;
 return currentRow;
 }
 var newTable = document.getElementById(
 genState.table_base_id + (genState.chunk_id + 1));
 if (newTable) {
 genState.chunk_id++;
 genState.row_num = 0;
 genState.table = newTable;
 return genState.table.rows[0];
 }
 return null;
 }
 var MAX_ROWS_PER_PASS = 1000;
 function continueToSetLineNumberHeights() {
 var rowsInThisPass = 0;
 var numRow = 1;
 var srcRow = 1;
 while (numRow && srcRow && rowsInThisPass < MAX_ROWS_PER_PASS) {
 numRow = rowGenerator(numsGenState);
 srcRow = rowGenerator(srcGenState);
 rowsInThisPass++;
 if (numRow && srcRow) {
 if (numRow.offsetHeight != srcRow.offsetHeight) {
 numRow.firstChild.style.height = srcRow.offsetHeight + 'px';
 }
 }
 }
 if (rowsInThisPass >= MAX_ROWS_PER_PASS) {
 setTimeout(continueToSetLineNumberHeights, 10);
 } else {
 alignerRunning = false;
 if (startOver) {
 startOver = false;
 setTimeout(setLineNumberHeights, 500);
 }
 }
 }
 // Do 2 complete passes, because there can be races
 // between this code and prettify.
 startOver = true;
 setTimeout(setLineNumberHeights, 250);
 window.onresize = setLineNumberHeights;
</script>

 
 
 <div id="log">
 <div style="text-align:right">
 <a class="ifCollapse" href="#" onclick="_toggleMeta('', 'p', 'numexpr', this)">Show details</a>
 <a class="ifExpand" href="#" onclick="_toggleMeta('', 'p', 'numexpr', this)">Hide details</a>
 </div>
 <div class="ifExpand">
 
 <div class="pmeta_bubble_bg" style="border:1px solid white">
 <div class="round4"></div>
 <div class="round2"></div>
 <div class="round1"></div>
 <div class="box-inner">
 <div id="changelog">
 <p>Change log</p>
 <div>
 <a href="/p/numexpr/source/detail?spec=svn234&r=204">r204</a>
 by faltet
 on Jul 30 (5 days ago)
 &nbsp; <a href="/p/numexpr/source/diff?spec=svn234&r=204&amp;format=side&amp;path=/trunk/numexpr/win32/pthread.h&amp;old_path=/branches/multithread/numexpr/win32/pthread.h&amp;old=194">Diff</a>
 </div>
 <pre>Merged branch multithreaded into trunk.</pre>
 </div>
 
 
 
 
 
 
 <script type="text/javascript">
 var detail_url = '/p/numexpr/source/detail?r=204&spec=svn234';
 var publish_url = '/p/numexpr/source/detail?r=204&spec=svn234#publish';
 // describe the paths of this revision in javascript.
 var changed_paths = [];
 var changed_urls = [];
 
 changed_paths.push('/trunk');
 changed_urls.push('/p/numexpr/source/browse/trunk?r=204&spec=svn234');
 
 
 changed_paths.push('/trunk/README.txt');
 changed_urls.push('/p/numexpr/source/browse/trunk/README.txt?r=204&spec=svn234');
 
 
 changed_paths.push('/trunk/RELEASE_NOTES.txt');
 changed_urls.push('/p/numexpr/source/browse/trunk/RELEASE_NOTES.txt?r=204&spec=svn234');
 
 
 changed_paths.push('/trunk/bench/poly.c');
 changed_urls.push('/p/numexpr/source/browse/trunk/bench/poly.c?r=204&spec=svn234');
 
 
 changed_paths.push('/trunk/bench/poly.py');
 changed_urls.push('/p/numexpr/source/browse/trunk/bench/poly.py?r=204&spec=svn234');
 
 
 changed_paths.push('/trunk/bench/unaligned-simple.py');
 changed_urls.push('/p/numexpr/source/browse/trunk/bench/unaligned-simple.py?r=204&spec=svn234');
 
 
 changed_paths.push('/trunk/numexpr/__init__.py');
 changed_urls.push('/p/numexpr/source/browse/trunk/numexpr/__init__.py?r=204&spec=svn234');
 
 
 changed_paths.push('/trunk/numexpr/interp_body.c');
 changed_urls.push('/p/numexpr/source/browse/trunk/numexpr/interp_body.c?r=204&spec=svn234');
 
 
 changed_paths.push('/trunk/numexpr/interpreter.c');
 changed_urls.push('/p/numexpr/source/browse/trunk/numexpr/interpreter.c?r=204&spec=svn234');
 
 
 changed_paths.push('/trunk/numexpr/tests/test_numexpr.py');
 changed_urls.push('/p/numexpr/source/browse/trunk/numexpr/tests/test_numexpr.py?r=204&spec=svn234');
 
 
 changed_paths.push('/trunk/numexpr/utils.py');
 changed_urls.push('/p/numexpr/source/browse/trunk/numexpr/utils.py?r=204&spec=svn234');
 
 
 changed_paths.push('/trunk/numexpr/win32');
 changed_urls.push('/p/numexpr/source/browse/trunk/numexpr/win32?r=204&spec=svn234');
 
 
 changed_paths.push('/trunk/numexpr/win32/pthread.c');
 changed_urls.push('/p/numexpr/source/browse/trunk/numexpr/win32/pthread.c?r=204&spec=svn234');
 
 
 changed_paths.push('/trunk/numexpr/win32/pthread.h');
 changed_urls.push('/p/numexpr/source/browse/trunk/numexpr/win32/pthread.h?r=204&spec=svn234');
 
 var selected_path = '/trunk/numexpr/win32/pthread.h';
 
 
 changed_paths.push('/trunk/setup.py');
 changed_urls.push('/p/numexpr/source/browse/trunk/setup.py?r=204&spec=svn234');
 
 
 function getCurrentPageIndex() {
 for (var i = 0; i < changed_paths.length; i++) {
 if (selected_path == changed_paths[i]) {
 return i;
 }
 }
 }
 function getNextPage() {
 var i = getCurrentPageIndex();
 if (i < changed_paths.length - 1) {
 return changed_urls[i + 1];
 }
 return null;
 }
 function getPreviousPage() {
 var i = getCurrentPageIndex();
 if (i > 0) {
 return changed_urls[i - 1];
 }
 return null;
 }
 function gotoNextPage() {
 var page = getNextPage();
 if (!page) {
 page = detail_url;
 }
 window.location = page;
 }
 function gotoPreviousPage() {
 var page = getPreviousPage();
 if (!page) {
 page = detail_url;
 }
 window.location = page;
 }
 function gotoDetailPage() {
 window.location = detail_url;
 }
 function gotoPublishPage() {
 window.location = publish_url;
 }
</script>
 
 <style type="text/css">
 #review_nav {
 border-top: 3px solid white;
 padding-top: 6px;
 margin-top: 1em;
 }
 #review_nav td {
 vertical-align: middle;
 }
 #review_nav select {
 margin: .5em 0;
 }
 </style>
 <div id="review_nav">
 <table><tr><td>Go to:&nbsp;</td><td>
 <select name="files_in_rev" onchange="window.location=this.value">
 
 <option value="/p/numexpr/source/browse/trunk?r=204&amp;spec=svn234"
 
 >/trunk</option>
 
 <option value="/p/numexpr/source/browse/trunk/README.txt?r=204&amp;spec=svn234"
 
 >/trunk/README.txt</option>
 
 <option value="/p/numexpr/source/browse/trunk/RELEASE_NOTES.txt?r=204&amp;spec=svn234"
 
 >/trunk/RELEASE_NOTES.txt</option>
 
 <option value="/p/numexpr/source/browse/trunk/bench/poly.c?r=204&amp;spec=svn234"
 
 >/trunk/bench/poly.c</option>
 
 <option value="/p/numexpr/source/browse/trunk/bench/poly.py?r=204&amp;spec=svn234"
 
 >/trunk/bench/poly.py</option>
 
 <option value="/p/numexpr/source/browse/trunk/bench/unaligned-simple.py?r=204&amp;spec=svn234"
 
 >/trunk/bench/unaligned-simple.py</option>
 
 <option value="/p/numexpr/source/browse/trunk/numexpr/__init__.py?r=204&amp;spec=svn234"
 
 >/trunk/numexpr/__init__.py</option>
 
 <option value="/p/numexpr/source/browse/trunk/numexpr/interp_body.c?r=204&amp;spec=svn234"
 
 >/trunk/numexpr/interp_body.c</option>
 
 <option value="/p/numexpr/source/browse/trunk/numexpr/interpreter.c?r=204&amp;spec=svn234"
 
 >/trunk/numexpr/interpreter.c</option>
 
 <option value="/p/numexpr/source/browse/trunk/numexpr/tests/test_numexpr.py?r=204&amp;spec=svn234"
 
 >...nk/numexpr/tests/test_numexpr.py</option>
 
 <option value="/p/numexpr/source/browse/trunk/numexpr/utils.py?r=204&amp;spec=svn234"
 
 >/trunk/numexpr/utils.py</option>
 
 <option value="/p/numexpr/source/browse/trunk/numexpr/win32?r=204&amp;spec=svn234"
 
 >/trunk/numexpr/win32</option>
 
 <option value="/p/numexpr/source/browse/trunk/numexpr/win32/pthread.c?r=204&amp;spec=svn234"
 
 >/trunk/numexpr/win32/pthread.c</option>
 
 <option value="/p/numexpr/source/browse/trunk/numexpr/win32/pthread.h?r=204&amp;spec=svn234"
 selected="selected"
 >/trunk/numexpr/win32/pthread.h</option>
 
 <option value="/p/numexpr/source/browse/trunk/setup.py?r=204&amp;spec=svn234"
 
 >/trunk/setup.py</option>
 
 </select>
 </td></tr></table>
 
 
 



 <div style="white-space:nowrap">
 
 <a href="https://www.google.com/accounts/ServiceLogin?service=code&amp;ltmpl=phosting&amp;continue=http%3A%2F%2Fcode.google.com%2Fp%2Fnumexpr%2Fsource%2Fbrowse%2Ftrunk%2Fnumexpr%2Fwin32%2Fpthread.h&amp;followup=http%3A%2F%2Fcode.google.com%2Fp%2Fnumexpr%2Fsource%2Fbrowse%2Ftrunk%2Fnumexpr%2Fwin32%2Fpthread.h"
 >Sign in</a> to write a code review</div>


 
 </div>
 
 
 </div>
 <div class="round1"></div>
 <div class="round2"></div>
 <div class="round4"></div>
 </div>
 <div class="pmeta_bubble_bg" style="border:1px solid white">
 <div class="round4"></div>
 <div class="round2"></div>
 <div class="round1"></div>
 <div class="box-inner">
 <div id="older_bubble">
 <p>Older revisions</p>
 
 
 <div class="closed" style="margin-bottom:3px;" >
 <img class="ifClosed" onclick="_toggleHidden(this)" src="http://www.gstatic.com/codesite/ph/images/plus.gif" >
 <img class="ifOpened" onclick="_toggleHidden(this)" src="http://www.gstatic.com/codesite/ph/images/minus.gif" >
 <a href="/p/numexpr/source/detail?spec=svn234&r=194">r194</a>
 by faltet
 on Jul 29 (6 days ago)
 &nbsp; <a href="/p/numexpr/source/diff?spec=svn234&r=194&amp;format=side&amp;path=/branches/multithread/numexpr/win32/pthread.h&amp;old_path=/branches/multithread/numexpr/win32/pthread.h&amp;old=0">Diff</a>
 <br>
 <pre class="ifOpened">Attempt to avoid the use of the
pthreads library on Win32.  Using the
pthreads API emulation for Win that
uses Git.</pre>
 </div>
 
 
 <a href="/p/numexpr/source/list?path=/trunk/numexpr/win32/pthread.h&start=204">All revisions of this file</a>
 </div>
 </div>
 <div class="round1"></div>
 <div class="round2"></div>
 <div class="round4"></div>
 </div>
 <div class="pmeta_bubble_bg" style="border:1px solid white">
 <div class="round4"></div>
 <div class="round2"></div>
 <div class="round1"></div>
 <div class="box-inner">
 <div id="fileinfo_bubble">
 <p>File info</p>
 
 <div>Size: 1888 bytes,
 68 lines</div>
 
 <div><a href="http://numexpr.googlecode.com/svn/trunk/numexpr/win32/pthread.h">View raw file</a></div>
 </div>
 
 </div>
 <div class="round1"></div>
 <div class="round2"></div>
 <div class="round4"></div>
 </div>
 </div>
 </div>


</div>
</div>

 <script src="http://www.gstatic.com/codesite/ph/7642550995449508181/js/prettify/prettify.js"></script>

<script type="text/javascript">prettyPrint();</script>

<script src="http://www.gstatic.com/codesite/ph/7642550995449508181/js/source_file_scripts.js"></script>

 <script type="text/javascript" src="http://kibbles.googlecode.com/files/kibbles-1.3.1.comp.js"></script>
 <script type="text/javascript">
 var lastStop = null;
 var initilized = false;
 
 function updateCursor(next, prev) {
 if (prev && prev.element) {
 prev.element.className = 'cursor_stop cursor_hidden';
 }
 if (next && next.element) {
 next.element.className = 'cursor_stop cursor';
 lastStop = next.index;
 }
 }
 
 function pubRevealed(data) {
 updateCursorForCell(data.cellId, 'cursor_stop cursor_hidden');
 if (initilized) {
 reloadCursors();
 }
 }
 
 function draftRevealed(data) {
 updateCursorForCell(data.cellId, 'cursor_stop cursor_hidden');
 if (initilized) {
 reloadCursors();
 }
 }
 
 function draftDestroyed(data) {
 updateCursorForCell(data.cellId, 'nocursor');
 if (initilized) {
 reloadCursors();
 }
 }
 function reloadCursors() {
 kibbles.skipper.reset();
 loadCursors();
 if (lastStop != null) {
 kibbles.skipper.setCurrentStop(lastStop);
 }
 }
 // possibly the simplest way to insert any newly added comments
 // is to update the class of the corresponding cursor row,
 // then refresh the entire list of rows.
 function updateCursorForCell(cellId, className) {
 var cell = document.getElementById(cellId);
 // we have to go two rows back to find the cursor location
 var row = getPreviousElement(cell.parentNode);
 row.className = className;
 }
 // returns the previous element, ignores text nodes.
 function getPreviousElement(e) {
 var element = e.previousSibling;
 if (element.nodeType == 3) {
 element = element.previousSibling;
 }
 if (element && element.tagName) {
 return element;
 }
 }
 function loadCursors() {
 // register our elements with skipper
 var elements = CR_getElements('*', 'cursor_stop');
 var len = elements.length;
 for (var i = 0; i < len; i++) {
 var element = elements[i]; 
 element.className = 'cursor_stop cursor_hidden';
 kibbles.skipper.append(element);
 }
 }
 function toggleComments() {
 CR_toggleCommentDisplay();
 reloadCursors();
 }
 function keysOnLoadHandler() {
 // setup skipper
 kibbles.skipper.addStopListener(
 kibbles.skipper.LISTENER_TYPE.PRE, updateCursor);
 // Set the 'offset' option to return the middle of the client area
 // an option can be a static value, or a callback
 kibbles.skipper.setOption('padding_top', 50);
 // Set the 'offset' option to return the middle of the client area
 // an option can be a static value, or a callback
 kibbles.skipper.setOption('padding_bottom', 100);
 // Register our keys
 kibbles.skipper.addFwdKey("n");
 kibbles.skipper.addRevKey("p");
 kibbles.keys.addKeyPressListener(
 'u', function() { window.location = detail_url; });
 kibbles.keys.addKeyPressListener(
 'r', function() { window.location = detail_url + '#publish'; });
 
 kibbles.keys.addKeyPressListener('j', gotoNextPage);
 kibbles.keys.addKeyPressListener('k', gotoPreviousPage);
 
 
 }
 window.onload = function() {keysOnLoadHandler();};
 </script>

<!-- code review support -->
<script src="http://www.gstatic.com/codesite/ph/7642550995449508181/js/code_review_scripts.js"></script>
<script type="text/javascript">
 
 // the comment form template
 var form = '<div class="draft"><div class="header"><span class="title">Draft comment:</span></div>' +
 '<div class="body"><form onsubmit="return false;"><textarea id="$ID">$BODY</textarea><br>$ACTIONS</form></div>' +
 '</div>';
 // the comment "plate" template used for both draft and published comment "plates".
 var draft_comment = '<div class="draft" ondblclick="$ONDBLCLICK">' +
 '<div class="header"><span class="title">Draft comment:</span><span class="actions">$ACTIONS</span></div>' +
 '<pre id="$ID" class="body">$BODY</pre>' +
 '</div>';
 var published_comment = '<div class="published">' +
 '<div class="header"><span class="title"><a href="$PROFILE_URL">$AUTHOR:</a></span><div>' +
 '<pre id="$ID" class="body">$BODY</pre>' +
 '</div>';

 function showPublishInstructions() {
 var element = document.getElementById('review_instr');
 if (element) {
 element.className = 'opened';
 }
 }
 function revsOnLoadHandler() {
 // register our source container with the commenting code
 var paths = {'svn204': '/trunk/numexpr/win32/pthread.h'}
 CR_setup('', 'p', 'numexpr', '', 'svn234', paths,
 '', CR_BrowseIntegrationFactory);
 // register our hidden ui elements with the code commenting code ui builder.
 CR_registerLayoutElement('form', form);
 CR_registerLayoutElement('draft_comment', draft_comment);
 CR_registerLayoutElement('published_comment', published_comment);
 
 CR_registerActivityListener(CR_ACTIVITY_TYPE.REVEAL_DRAFT_PLATE, showPublishInstructions);
 
 CR_registerActivityListener(CR_ACTIVITY_TYPE.REVEAL_PUB_PLATE, pubRevealed);
 CR_registerActivityListener(CR_ACTIVITY_TYPE.REVEAL_DRAFT_PLATE, draftRevealed);
 CR_registerActivityListener(CR_ACTIVITY_TYPE.DISCARD_DRAFT_COMMENT, draftDestroyed);
 
 
 
 
 
 
 
 
 
 var initilized = true;
 reloadCursors();
 }
 window.onload = function() {keysOnLoadHandler(); revsOnLoadHandler();};
</script>
<script type="text/javascript" src="http://www.gstatic.com/codesite/ph/7642550995449508181/js/dit_scripts.js"></script>

 
 
 <script type="text/javascript" src="http://www.gstatic.com/codesite/ph/7642550995449508181/js/core_scripts_20081103.js"></script>
 <script type="text/javascript" src="/js/codesite_product_dictionary_ph.pack.04102009.js"></script>
 </div>
<div id="footer" dir="ltr">
 
 <div class="text">
 
 &copy;2010 Google -
 <a href="/projecthosting/terms.html">Terms</a> -
 <a href="http://www.google.com/privacy.html">Privacy</a> -
 <a href="/p/support/">Project Hosting Help</a>
 
 </div>
</div>

 <div class="hostedBy" style="margin-top: -20px;">
 <span style="vertical-align: top;">Powered by <a href="http://code.google.com/projecthosting/">Google Project Hosting</a></span>
 </div>
 
 


 
 </body>
</html>

